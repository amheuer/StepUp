"""Main game loop and rendering."""

import sys
import random
import math
import cv2
from pathlib import Path
import json

import pygame
import config as cfg

from cv_tester import CVController, CVState
from collisions import check_if_on_platform
from coins import Coin, collect_coins, cull_coins, spawn_coins_near_platforms
from config import (
	COLOR_BARS,
	COLOR_BARS_PATTERN,
	COLOR_BARS_TEXT,
	COLOR_GAME_OVER,
	COLOR_TEXT,
	CV_MOVE_X_SCALE,
	CV_MOVE_DEADZONE,
	CV_JUMP_SPEED_SCALE,
	FPS,
	SCROLL_THRESHOLD,
	UI_FONT_SIZE,
	WINDOW_HEIGHT,
	WINDOW_TITLE,
	WINDOW_WIDTH,
	COIN_RADIUS,
	COIN_VALUE,
	PLAYER_RADIUS,
	PLAYER_MAX_X_SPEED,
	PLAYER_MAX_SPEED,
	PLAYER_HEIGHT_METERS,
	PLATFORM_IGNORE_TIME,
	PLATFORM_HEIGHT,
	JUMP_REARM_TIME,
)
from entities import Player, Platform
from platforms import create_initial_platforms, generate_new_platforms
from yolo_sam_segment import user_has_photos, capture_and_process_for_user

BG_DIR = (
	Path(__file__).resolve().parent
	/ "assets"
	/ "Extraordinary Pixelvania - Free Asset Pack"
	/ "Sprites"
	/ "BGs"
)
FONT_PATH = (
	Path(__file__).resolve().parent
	/ "assets"
	/ "Extraordinary Pixelvania - Free Asset Pack"
	/ "Font"
	/ "Extraordinary Font.ttf"
)
SOUNDTRACK_PATH = Path(__file__).resolve().parent / "assets" / "Sounds" / "Jeremy Blake - Powerup!.mp3"
SFX_DIR = Path(__file__).resolve().parent / "assets" / "brackeys_platformer_assets" / "sounds"
COIN_DIR = Path(__file__).resolve().parent / "assets" / "Coin" / "spinning_coin"
TILESET_PATH = (
	Path(__file__).resolve().parent
	/ "assets"
	/ "Extraordinary Pixelvania - Free Asset Pack"
	/ "Sprites"
	/ "Tileset"
	/ "Tileset.png"
)
USERS_PATH = Path(__file__).resolve().parent / "users.json"
PLAYER_DIR = Path(__file__).resolve().parent / "assets" / "player_images"


class Game:
	"""Main game manager."""

	def __init__(self):
		pygame.init()
		pygame.mixer.init()
		self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
		pygame.display.set_caption(WINDOW_TITLE)
		self.clock = pygame.time.Clock()
		self.ui_font = pygame.font.Font(FONT_PATH, UI_FONT_SIZE)
		self.font = self.ui_font
		self.title_font = pygame.font.Font(FONT_PATH, 14)
		self.subtitle_font = pygame.font.Font(FONT_PATH, 6)
		self.menu_font = pygame.font.Font(FONT_PATH, 8)
		self.game_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
		self.high_score = 0
		self.next_high_score_sfx = 1000
		self.music_volume = 0.7
		self.sfx_volume = 0.7
		tileset = pygame.image.load(TILESET_PATH).convert_alpha()
		Platform.load_tileset(tileset)
		self._load_player_sprites()
		self._load_coin_frames()
		self.sfx = self._load_sfx()
		pygame.mixer.music.load(SOUNDTRACK_PATH)
		pygame.mixer.music.set_volume(self.music_volume)
		pygame.mixer.music.play(-1)
		self.skip_sfx_frames = 0
		self._load_backgrounds()
		self.particle_surface_far = self._build_particle_layer(50)
		self.particle_surface_near = self._build_particle_layer(90)
		self.bar_pattern_surface = None
		self.bar_pattern_size = (0, 0)
		self.menu_pattern_surface = None
		self.menu_pattern_size = (0, 0)
		self.cv = None
		self.cv_state = CVState(
			zone=None,
			jump_active=False,
			jump_vector=None,
			center=None,
			center_velocity=None,
			bbox_center=None,
			debug_frame=None,
			timestamp=0.0,
			has_person=False,
		)
		self.countdown_active = False
		self.countdown_remaining = 0.0
		self.landing_stick_timer = 0.0
		self.landing_stick_duration = 0.0
		self.last_jump_vector = None
		self.prev_jump_active = False
		self.jump_flash_timer = 0.0
		self.last_jump_triggered = False
		self.prev_game_over = False

		self.reset()
		self.running = True
		self.game_over = False
		self.paused = False
		self.pause_index = 0
		self.pause_options = ["RESUME", "MUSIC", "SFX", "CONTROL", "INTENSITY", "RESTART", "MAIN MENU", "QUIT"]
		self.in_menu = True
		self.in_health = False
		self.in_tutorial = False
		self.portal_rect = None
		self.portal_timer = 0.0
		self.menu_index = 0
		self.menu_options = ["PLAY", "HEALTH  INFO", "SIGN IN"]
		self.menu_option_rects = {}
		self.signin_rect = None
		self.control_mode = "CV"
		self.pause_option_rects = {}
		self.render_scale = 1.0
		self.render_offset = (0, 0)
		self.hover_menu_index = None
		self.hover_pause_index = None
		self.signin_mode = False
		self.username_input = ""
		self.current_user = None
		self.users = self._load_users()
		self.user_save_timer = 0.0
		self._init_cv()

	def _init_cv(self):
		try:
			self.cv = CVController(debug_draw=True, show_window=False)
			self.cv.start()
		except Exception as exc:
			self.cv = None
			print(f"[CV] Failed to initialize CV controller: {exc}")

	def reset(self):
		"""Reset game state for a new game."""
		self.player = Player(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 80)
		self.platforms = create_initial_platforms()
		self.coins = spawn_coins_near_platforms(self.platforms)
		self.score = 0
		self.height_jumped = 0.0
		self.coins_collected = 0
		self.calories = 0.0
		self.display_calories = 0.0
		self.next_high_score_sfx = 1000
		self.game_over = False
		self.paused = False
		pygame.mixer.music.play(-1)
		self.skip_sfx_frames = 2
		self.countdown_active = True
		self.countdown_remaining = 3.0
		self.landing_stick_timer = 0.0
		self.last_jump_vector = None
		self.prev_jump_active = False
		self.jump_flash_timer = 0.0
		self.last_jump_triggered = False
		self.prev_game_over = False
		self._reset_balance_tracking()

	def _update_calories(self):
		"""Return calories per minute based on intensity."""
		return cfg.INTENSITY.value

	def _load_users(self):
		if not USERS_PATH.exists():
			return {}
		try:
			return json.loads(USERS_PATH.read_text())
		except Exception:
			return {}

	def _save_users(self):
		try:
			USERS_PATH.write_text(json.dumps(self.users, indent=2))
		except Exception:
			pass

	def _ensure_user(self, username):
		key = username.strip().upper()
		if key not in self.users:
			self.users[key] = {
				"highscore": 0,
				"lifetime_calories": 0.0,
				"minutes_played": 0.0,
				"balance_ability": 0.0,
			}
		self.current_user = key

	def _reset_balance_tracking(self):
		self.balance_jump_v_sum = 0.0
		self.balance_jump_t_sum = 0.0
		self.balance_shuffle_v_sum = 0.0
		self.balance_shuffle_t_sum = 0.0
		self.balance_moves = 0
		self.balance_time_no_moves = 0.0
		self.balance_jump_in_progress = False
		self.balance_shuffle_in_progress = False

	def _update_balance_tracking(self, dt):
		if self.in_tutorial:
			return
		has_vel = self.cv_state.center_velocity is not None
		vx = self.cv_state.center_velocity[0] if has_vel else 0.0
		vy = self.cv_state.center_velocity[1] if has_vel else 0.0

		if self.balance_jump_in_progress:
			if (not self.cv_state.jump_active) and (not has_vel or abs(vy) < CV_MOVE_DEADZONE):
				self.balance_jump_in_progress = False
		else:
			if self.cv_state.jump_active:
				self.balance_jump_in_progress = True
				self.balance_moves += 1

		shuffle_active = has_vel and abs(vx) >= CV_MOVE_DEADZONE
		if shuffle_active and not self.balance_shuffle_in_progress:
			self.balance_shuffle_in_progress = True
			self.balance_moves += 1
		elif not shuffle_active and self.balance_shuffle_in_progress:
			self.balance_shuffle_in_progress = False

		if self.balance_jump_in_progress and has_vel:
			self.balance_jump_v_sum += abs(vy) * dt
			self.balance_jump_t_sum += dt
		if self.balance_shuffle_in_progress and has_vel:
			self.balance_shuffle_v_sum += abs(vx) * dt
			self.balance_shuffle_t_sum += dt
		if not self.balance_jump_in_progress and not self.balance_shuffle_in_progress:
			self.balance_time_no_moves += dt

	def _compute_balance_ability(self):
		if self.balance_moves <= 0:
			return 0.0
		transition_time = self.balance_time_no_moves / self.balance_moves
		if transition_time <= 0.0:
			return 0.0
		vv = 0.0 if self.balance_jump_t_sum <= 0 else (self.balance_jump_v_sum / self.balance_jump_t_sum)
		vh = 0.0 if self.balance_shuffle_t_sum <= 0 else (self.balance_shuffle_v_sum / self.balance_shuffle_t_sum)
		return (vh + vv) / transition_time

	def _finalize_balance_ability(self):
		if not self.current_user:
			return
		user = self.users.get(self.current_user)
		if not user:
			return
		user["balance_ability"] = self._compute_balance_ability()
		self._save_users()

	def _user_sprite_dir(self, username=None):
		"""Return the per-user sprite directory path."""
		name = username or self.current_user
		if not name:
			return None
		return PLAYER_DIR / name.strip().upper()

	def _ensure_user_photos(self):
		"""If the current user has no photos, run the capture pipeline.

		Minimises pygame while the webcam / CV windows are active,
		then restores and reloads the sprites.  Returns True if sprites
		are ready (either already existed or were just created).
		"""
		if not self.current_user:
			return True  # no user signed in – use default sprites

		sprite_dir = self._user_sprite_dir()
		if user_has_photos(str(sprite_dir)):
			# Already have photos – just make sure they're loaded
			self._load_player_sprites(sprite_dir)
			return True

		# Stop the CV controller so its webcam is released for capture
		if self.cv:
			self.cv.stop()
			self.cv = None

		# Minimise pygame so the OpenCV windows are visible
		pygame.display.iconify()

		try:
			success = capture_and_process_for_user(
				player_dir=str(sprite_dir),
				countdown=10,
			)
		except Exception as exc:
			print(f"[WARN] Photo capture failed: {exc}")
			success = False

		# Restore the pygame window *behind* the still-open black CV window
		self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
		pygame.display.flip()          # make sure the surface is
		pygame.event.pump()            # presented before we remove the cover

		# Now destroy the CV overlay so the transition is seamless
		cv2.destroyAllWindows()

		# Restart the CV controller
		self._init_cv()

		if success and user_has_photos(str(sprite_dir)):
			self._load_player_sprites(sprite_dir)
			return True
		else:
			print("[INFO] Using default sprites.")
			self._load_player_sprites()
			return False

	def _load_sfx(self):
		"""Load sound effects by filename."""
		names = [
			"coin.wav",
			"explosion.wav",
			"hurt.wav",
			"jump.wav",
			"power_up.wav",
			"tap.wav",
		]
		sfx = {}
		for name in names:
			path = SFX_DIR / name
			sound = pygame.mixer.Sound(path)
			if name == "coin.wav":
				sound.set_volume(0.35)
			else:
				sound.set_volume(self.sfx_volume)
			sfx[name] = sound
		return sfx

	def _load_coin_frames(self):
		"""Load spinning coin animation frames."""
		frames = []
		for i in range(1, 11):
			path = COIN_DIR / f"coin{i}.png"
			frame = pygame.image.load(path).convert_alpha()
			size = COIN_RADIUS * 2
			frame = pygame.transform.smoothscale(frame, (size, size))
			frames.append(frame)
		Coin.load_frames(frames, frame_time_ms=80)

	def _load_player_sprites(self, sprite_dir=None):
		"""Load player sprites from the given directory, or the default."""
		src = Path(sprite_dir) if sprite_dir else PLAYER_DIR
		stand = pygame.image.load(src / "stand_still.png").convert_alpha()
		left = pygame.image.load(src / "move_left.png").convert_alpha()
		jump = pygame.image.load(src / "jump.png").convert_alpha()

		target_w = PLAYER_RADIUS * 2
		def scale_by_width(img):
			h = int(img.get_height() * (target_w / img.get_width()))
			return pygame.transform.smoothscale(img, (target_w, h))

		stand = scale_by_width(stand)
		left = scale_by_width(left)
		jump = scale_by_width(jump)
		# Match heights to stand while preserving aspect ratio.
		def match_height(img, target_h):
			w = int(img.get_width() * (target_h / img.get_height()))
			return pygame.transform.smoothscale(img, (w, target_h))
		left = match_height(left, stand.get_height())
		right = pygame.transform.flip(left, True, False)
		Player.load_sprites(stand, left, right, jump)

	def _load_backgrounds(self):
		"""Load background layers from the asset pack."""
		bg1_path = BG_DIR / "bg1.png"
		bg2_path = BG_DIR / "bg2.png"
		p1_path = BG_DIR / "bg_particle_1.png"
		p2_path = BG_DIR / "bg_particle_2.png"
		self.bg_base = pygame.image.load(bg1_path).convert()
		self.bg_bottom = pygame.image.load(bg2_path).convert_alpha()
		self.particle_1 = pygame.image.load(p1_path).convert_alpha()
		self.particle_2 = pygame.image.load(p2_path).convert_alpha()

	def _build_particle_layer(self, count):
		"""Pre-render a random particle cloud background."""
		surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
		for i in range(count):
			sprite = self.particle_1 if i % 2 == 0 else self.particle_2
			max_x = max(0, WINDOW_WIDTH - sprite.get_width())
			max_y = max(0, WINDOW_HEIGHT - sprite.get_height())
			x = random.randint(0, max_x)
			y = random.randint(0, max_y)
			surf.blit(sprite, (x, y))
		return surf

	def _blit_vertical_tiled(self, surface, offset_y):
		"""Blit a surface vertically tiled with the given offset."""
		height = surface.get_height()
		if height <= 0:
			return
		offset = offset_y % height
		self.game_surface.blit(surface, (0, -offset))
		self.game_surface.blit(surface, (0, -offset + height))

	def _screen_to_game(self, pos):
		"""Convert screen coordinates to game-surface coordinates."""
		ox, oy = self.render_offset
		if self.render_scale <= 0:
			return None
		gx = (pos[0] - ox) / self.render_scale
		gy = (pos[1] - oy) / self.render_scale
		if gx < 0 or gy < 0 or gx > WINDOW_WIDTH or gy > WINDOW_HEIGHT:
			return None
		return (gx, gy)
	
	def _map_jump_vector(self, cv_vector):
		if not cv_vector:
			return None
		vx, vy = cv_vector
		speed = math.hypot(vx, vy)
		if speed <= 1e-6:
			return None
		nx = vx / speed
		ny = vy / speed
		# Rescale angle from +/-90 around vertical to +/-35 around vertical.
		theta = math.atan2(nx, -ny)
		clamped = max(-math.pi / 2, min(math.pi / 2, theta))
		scaled = (clamped / (math.pi / 2)) * math.radians(35)
		sin_t = math.sin(scaled)
		cos_t = math.cos(scaled)
		return (sin_t, -cos_t)

	def _build_bar_pattern(self, width, height):
		"""Create a subtle square pattern for the side bars."""
		surf = pygame.Surface((width, height))
		surf.fill(COLOR_BARS)
		rng = random.Random(1337)
		tile = 16
		for y in range(0, height, tile):
			for x in range(0, width, tile):
				if rng.random() < 0.35:
					pygame.draw.rect(surf, COLOR_BARS_PATTERN, (x, y, tile, tile))
		return surf

	def _build_menu_pattern(self, width, height):
		"""Create a random pixel pattern for the menu background."""
		surf = pygame.Surface((width, height))
		surf.fill(COLOR_BARS)
		rng = random.Random(2024)
		tile = 8
		for y in range(0, height, tile):
			for x in range(0, width, tile):
				if rng.random() < 0.55:
					pygame.draw.rect(surf, COLOR_BARS_PATTERN, (x, y, tile, tile))
		return surf

	# ── Tutorial level ──────────────────────────────────────────────
	def _setup_tutorial(self):
		"""Set up the tutorial level layout.

		Layout (all coordinates in the 400×600 game surface):
		  - Solid ground spanning the full width at the bottom.
		  - A platform on the right side, a bit above the ground.
		  - A platform on the left side, higher up.
		  - A portal centred at the top of the screen.
		"""
		self.in_tutorial = True
		self.portal_timer = 0.0

		# Start with an empty platform list
		self.platforms = []

		# 1.  Solid ground — a wide platform that never fades
		ground = Platform(0, WINDOW_HEIGHT - 30,
						  width=WINDOW_WIDTH, height=30, kind="normal")
		ground.fade_duration = 1e9          # effectively never fades
		self.platforms.append(ground)

		# 2.  Right-side step
		step_w = 90
		right_plat = Platform(
			WINDOW_WIDTH - step_w - 30, WINDOW_HEIGHT - 160,
			width=step_w, height=PLATFORM_HEIGHT, kind="normal",
		)
		right_plat.fade_duration = 1e9
		self.platforms.append(right_plat)

		# 3.  Left-side step (higher)
		left_plat = Platform(
			30, WINDOW_HEIGHT - 300,
			width=step_w, height=PLATFORM_HEIGHT, kind="normal",
		)
		left_plat.fade_duration = 1e9
		self.platforms.append(left_plat)

		# 4.  Portal at the top centre
		portal_w, portal_h = 50, 60
		self.portal_rect = pygame.Rect(
			(WINDOW_WIDTH - portal_w) // 2,
			30,
			portal_w,
			portal_h,
		)

		# Place the player on the ground
		self.player = Player(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30 - PLAYER_RADIUS)
		self.coins = []
		self.countdown_active = False  # No countdown in tutorial

	def _exit_tutorial(self):
		"""Leave the tutorial and start the real game."""
		self.in_tutorial = False
		self.portal_rect = None
		self.reset()

	def _draw_portal(self):
		"""Draw a swirling portal effect at self.portal_rect."""
		if self.portal_rect is None:
			return
		cx = self.portal_rect.centerx
		cy = self.portal_rect.centery
		t = self.portal_timer
		# Concentric pulsing ellipses
		for i in range(4, 0, -1):
			pulse = 1.0 + 0.15 * math.sin(t * 3 + i)
			rw = int(self.portal_rect.w // 2 * (i / 4) * pulse)
			rh = int(self.portal_rect.h // 2 * (i / 4) * pulse)
			alpha = 120 + 30 * i
			hue_shift = int((t * 80 + i * 40) % 256)
			color = pygame.Color(0)
			color.hsva = (hue_shift % 360, 80, 100, 0)
			r, g, b = color.r, color.g, color.b
			surf = pygame.Surface((rw * 2, rh * 2), pygame.SRCALPHA)
			pygame.draw.ellipse(surf, (r, g, b, alpha), (0, 0, rw * 2, rh * 2))
			self.game_surface.blit(surf, (cx - rw, cy - rh))

	def _start_game(self):
		"""Handle the PLAY action: capture photos if needed, then start."""
		# Check whether character creation is about to happen
		needs_creation = False
		if self.current_user:
			sprite_dir = self._user_sprite_dir()
			if not user_has_photos(str(sprite_dir)):
				needs_creation = True

		self._ensure_user_photos()
		self.reset()

		# Only show the tutorial right after character creation
		if needs_creation:
			self._setup_tutorial()

		self.in_menu = False
		self.in_health = False

	def handle_events(self):
		"""Process input events."""
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self._save_users()
				self.running = False
			elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
				pos = event.pos
				if self.in_menu:
					for key, rect in self.menu_option_rects.items():
						if rect.collidepoint(pos):
							if key == "PLAY":
								self._start_game()
							elif key.startswith("HEALTH"):
								self.in_menu = False
								self.in_health = True
							elif key == "SIGN IN":
								self.signin_mode = True
							break
					continue
				if self.paused:
					game_pos = self._screen_to_game(pos)
					if not game_pos:
						continue
					gx, gy = game_pos
					for key, rect in self.pause_option_rects.items():
						if rect.collidepoint(gx, gy):
							if key == "RESUME":
								self.paused = False
							elif key == "RESTART":
								self.reset()
								self.game_over = False
								self.paused = False
							elif key == "MAIN MENU":
								self.paused = False
								self.in_health = False
								self.in_menu = True
							elif key == "QUIT":
								self.running = False
							break
				continue
			elif event.type == pygame.KEYDOWN:
				if self.signin_mode:
					if event.key == pygame.K_RETURN:
						name = self.username_input.strip().upper()
						if name:
							self._ensure_user(name)
							self.username_input = ""
							self.signin_mode = False
							self._save_users()
					elif event.key == pygame.K_ESCAPE:
						self.username_input = ""
						self.signin_mode = False
					elif event.key == pygame.K_BACKSPACE:
						self.username_input = self.username_input[:-1]
					else:
						char = event.unicode.upper()
						if char.isprintable() and len(self.username_input) < 16:
							self.username_input += char
					continue
				if self.in_menu:
					if event.key in (pygame.K_LEFT, pygame.K_a):
						self.menu_index = (self.menu_index - 1) % len(self.menu_options)
					elif event.key in (pygame.K_RIGHT, pygame.K_d):
						self.menu_index = (self.menu_index + 1) % len(self.menu_options)
					if event.key == pygame.K_TAB:
						self.signin_mode = True
						continue
					if event.key == pygame.K_i:
						self.signin_mode = True
						continue
					elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
						choice = self.menu_options[self.menu_index]
						if choice == "PLAY":
							self._start_game()
						elif choice.startswith("HEALTH"):
							self.in_menu = False
							self.in_health = True
						elif choice == "SIGN IN":
							self.signin_mode = True
					elif event.key == pygame.K_ESCAPE:
						self.running = False
					continue
				if self.in_health:
					if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE, pygame.K_RETURN, pygame.K_SPACE):
						self.in_health = False
						self.in_menu = True
					continue
				if event.key == pygame.K_ESCAPE:
					self.paused = not self.paused
				elif self.game_over and event.key == pygame.K_r:
					self.reset()
					self.game_over = False
				elif self.paused:
					if event.key in (pygame.K_UP, pygame.K_w):
						self.pause_index = (self.pause_index - 1) % len(self.pause_options)
					elif event.key in (pygame.K_DOWN, pygame.K_s):
						self.pause_index = (self.pause_index + 1) % len(self.pause_options)
					elif event.key in (pygame.K_LEFT, pygame.K_a):
						choice = self.pause_options[self.pause_index]
						if choice == "MUSIC":
							self.music_volume = max(0.0, self.music_volume - 0.1)
							pygame.mixer.music.set_volume(self.music_volume)
						elif choice == "SFX":
							self.sfx_volume = max(0.0, self.sfx_volume - 0.1)
							for s in self.sfx.values():
								s.set_volume(self.sfx_volume)
							self.sfx["coin.wav"].set_volume(min(0.35, self.sfx_volume))
						elif choice == "CONTROL":
							self.control_mode = "ARROWS" if self.control_mode == "CV" else "CV"
						elif choice == "INTENSITY":
							self._cycle_intensity(-1)
					elif event.key in (pygame.K_RIGHT, pygame.K_d):
						choice = self.pause_options[self.pause_index]
						if choice == "MUSIC":
							self.music_volume = min(1.0, self.music_volume + 0.1)
							pygame.mixer.music.set_volume(self.music_volume)
						elif choice == "SFX":
							self.sfx_volume = min(1.0, self.sfx_volume + 0.1)
							for s in self.sfx.values():
								s.set_volume(self.sfx_volume)
							self.sfx["coin.wav"].set_volume(min(0.35, self.sfx_volume))
						elif choice == "INTENSITY":
							self._cycle_intensity(1)
					elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
						choice = self.pause_options[self.pause_index]
						if choice == "RESUME":
							self.paused = False
						elif choice == "RESTART":
							self.reset()
							self.game_over = False
							self.paused = False
						elif choice == "MAIN MENU":
							self.paused = False
							self.in_health = False
							self.in_menu = True
						elif choice == "QUIT":
							self.running = False

	def update(self, dt):
		"""Update game state."""
		if self.game_over or self.paused or self.in_menu or self.in_health:
			return
		if self.countdown_active:
			self.countdown_remaining = max(0.0, self.countdown_remaining - dt)
			if self.countdown_remaining <= 0.0:
				self.countdown_active = False
			return

		# Store previous position for collision detection
		prev_y = self.player.y
		was_on_platform = self.player.on_platform
		prev_high_score = self.high_score
		play_sfx = self.skip_sfx_frames <= 0
		if self.skip_sfx_frames > 0:
			self.skip_sfx_frames -= 1
		keys = pygame.key.get_pressed()


		# Update CV state
		if self.cv:
			self.cv_state = self.cv.get_state()
		else:
			self.cv_state = CVState(
				zone=None,
				jump_active=False,
				jump_vector=None,
				center=None,
				center_velocity=None,
				bbox_center=None,
				debug_frame=None,
				timestamp=0.0,
				has_person=False,
			)

		if self.control_mode == "CV":
			if self.cv_state.jump_vector is not None:
				self.last_jump_vector = self.cv_state.jump_vector
			jump_vector = None
			if self.cv_state.jump_active and self.last_jump_vector is not None:
				jump_vector = self._map_jump_vector(self.last_jump_vector)
		else:
			jump_vector = None
		if self.player.ignore_platform_timer > 0:
			self.player.ignore_platform_timer = max(0.0, self.player.ignore_platform_timer - dt)

		move_vx = 0.0
		if self.control_mode == "CV":
			if self.cv_state.center_velocity is not None:
				move_vx = self.cv_state.center_velocity[0] * CV_MOVE_X_SCALE
				if abs(move_vx) < CV_MOVE_DEADZONE:
					move_vx = 0.0
			move_vx = max(-PLAYER_MAX_X_SPEED, min(PLAYER_MAX_X_SPEED, move_vx))
			jump_active = self.cv_state.jump_active
		else:
			if keys[pygame.K_LEFT] or keys[pygame.K_a]:
				move_vx = -PLAYER_MAX_X_SPEED
			if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
				move_vx = PLAYER_MAX_X_SPEED
			jump_active = keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_SPACE]

		self._update_balance_tracking(dt)

		jump_allowed = True

		if self.player.y < prev_y:
			pixels_up = (prev_y - self.player.y) / 2
			meters_per_pixel = PLAYER_HEIGHT_METERS / (PLAYER_RADIUS * 2)
			self.height_jumped += pixels_up * meters_per_pixel
		for platform in self.platforms:
			platform.update(dt)
			if play_sfx and platform.just_broke:
				self.sfx["hurt.wav"].play()

		# Reset jump ability each frame, only set if colliding with platform
		self.player.can_jump = False

		# Reset platform state each frame
		self.player.on_platform = False

		# Check if player is on a platform
		platform_below = None
		if self.player.ignore_platform_timer <= 0:
			platform_below = check_if_on_platform(self.player, self.platforms, prev_y)

		if platform_below:
			self.player.on_platform = True
			self.player.vx = 0
			self.player.vy = 0  # Stop velocity when on platform (gravity won't apply while on platform)
			self.player.y = platform_below.rect.y - self.player.radius
			if self.player.time_since_jump is None or self.player.time_since_jump >= JUMP_REARM_TIME:
				self.player.can_jump = True  # Allow player to jump
				if not jump_active:
					self.prev_jump_active = False
			self.player.x += platform_below.vx * dt
			min_x = platform_below.x + self.player.radius
			max_x = platform_below.x + platform_below.w - self.player.radius
			if max_x < min_x:
				max_x = min_x
			self.player.x = max(min_x, min(max_x, self.player.x))
			platform_below.land()  # Start fading this platform
			if play_sfx and not was_on_platform:
				self.sfx["tap.wav"].play()
			if not was_on_platform:
				self.display_calories = self.calories
				self.landing_stick_timer = self.landing_stick_duration
			if self.landing_stick_timer > 0:
				self.landing_stick_timer = max(0.0, self.landing_stick_timer - dt)
				move_dir = 0
				jump_allowed = False
		else:
			self.landing_stick_timer = 0.0

		jump_triggered = jump_active and not self.prev_jump_active
		self.last_jump_triggered = jump_triggered


		# Update entities
		self.player.update(dt, move_vx, jump_triggered and jump_allowed, jump_vector)
		if play_sfx and self.player.jumped_this_frame:
			self.sfx["jump.wav"].play()
			self.jump_flash_timer = 0.4
		if self.player.jumped_this_frame:
			self.player.ignore_platform_timer = PLATFORM_IGNORE_TIME
		if platform_below and self.player.on_platform and self.player.ignore_platform_timer <= 0:
			self.player.y = platform_below.rect.y - self.player.radius
			min_x = platform_below.x + self.player.radius
			max_x = platform_below.x + platform_below.w - self.player.radius
			if max_x < min_x:
				max_x = min_x
			self.player.x = max(min_x, min(max_x, self.player.x))

		# ── Tutorial-specific vs normal game logic ──────────────────
		if self.in_tutorial:
			# No scrolling in the tutorial
			# Keep player within the screen bounds
			if self.player.y - self.player.radius > WINDOW_HEIGHT:
				# Respawn on the ground instead of game-over
				self.player.x = WINDOW_WIDTH // 2
				self.player.y = WINDOW_HEIGHT - 30 - PLAYER_RADIUS
				self.player.vx = 0
				self.player.vy = 0

			# Animate portal
			self.portal_timer += dt

			# Check portal collision
			if self.portal_rect and self.player.rect.colliderect(self.portal_rect):
				if play_sfx:
					self.sfx["power_up.wav"].play()
				self._exit_tutorial()
				return
		else:
			# Handle scrolling
			if self.player.y < SCROLL_THRESHOLD:
				dy = SCROLL_THRESHOLD - self.player.y
				self.player.y = SCROLL_THRESHOLD
				for platform in self.platforms:
					platform.y += dy
				for coin in self.coins:
					coin.y += dy

			# Remove off-screen or faded platforms
			self.platforms = [
				p for p in self.platforms if p.y < WINDOW_HEIGHT + 50 and p.is_active
			]

			# Generate new platforms
			new_platforms = generate_new_platforms(self.platforms)

			# Add coins near platforms
			self.coins.extend(spawn_coins_near_platforms(new_platforms))

			# Collect coins
			coin_score, coin_count = collect_coins(self.player, self.coins)
			self.coins_collected += coin_count
			if play_sfx and coin_count > 0:
				self.sfx["coin.wav"].play()
			self.score = int(self.height_jumped * 2) + self.coins_collected * COIN_VALUE
			if self.score > self.high_score:
				self.high_score = self.score
			if self.current_user:
				user = self.users[self.current_user]
				if self.high_score > user["highscore"]:
					user["highscore"] = self.high_score
			if play_sfx and self.high_score >= self.next_high_score_sfx:
				self.sfx["power_up.wav"].play()
				self.next_high_score_sfx += 1000

			# Cull coins
			self.coins = cull_coins(self.coins)

			# Check game over condition
			if self.player.y - self.player.radius > WINDOW_HEIGHT:
				if play_sfx:
					self.sfx["explosion.wav"].play()
				pygame.mixer.music.stop()
				self.game_over = True

		per_min = self._update_calories()
		self.calories += per_min * (dt / 60.0)
		if self.player.jumped_this_frame:
			self.display_calories = self.calories
		if self.current_user:
			user = self.users[self.current_user]
			user["lifetime_calories"] += per_min * (dt / 60.0)
			user["minutes_played"] += dt / 60.0
			self.user_save_timer += dt
			if self.user_save_timer >= 5.0:
				self._save_users()
				self.user_save_timer = 0.0
		
		if self.jump_flash_timer > 0:
			self.jump_flash_timer = max(0.0, self.jump_flash_timer - dt)

		self.prev_jump_active = jump_active
		if self.game_over and not self.prev_game_over:
			self._finalize_balance_ability()
		self.prev_game_over = self.game_over

	def draw_background(self):
		"""Draw layered background with parallax bubbles."""
		base = pygame.transform.scale(self.bg_base, (WINDOW_WIDTH, WINDOW_HEIGHT))
		self.game_surface.blit(base, (0, 0))

		self._blit_vertical_tiled(self.particle_surface_far, int(-self.height_jumped * 0.03))
		self._blit_vertical_tiled(self.particle_surface_near, int(-self.height_jumped * 0.06))

		top_w = WINDOW_WIDTH
		scale = top_w / self.bg_bottom.get_width()
		top_h = int(self.bg_bottom.get_height() * scale)
		top = pygame.transform.scale(self.bg_bottom, (top_w, top_h))
		self.game_surface.blit(top, (0, WINDOW_HEIGHT - top_h))

	def draw_ui(self):
		"""Draw score and game over text."""
		if self.game_over:
			game_over_text = self.font.render(
				"GAME OVER - PRESS R TO RESTART",
				True,
				COLOR_GAME_OVER,
			)
			game_over_text = pygame.transform.smoothscale(
				game_over_text,
				(
					max(1, game_over_text.get_width() // 2.5),
					max(1, game_over_text.get_height() // 2.5),
				),
			)
			box_padding = 10
			box_w = game_over_text.get_width() + box_padding * 2
			box_h = game_over_text.get_height() + box_padding * 2
			box_x = (WINDOW_WIDTH - box_w) // 2
			box_y = WINDOW_HEIGHT // 2 - 20 - box_padding
			box_rect = pygame.Rect(box_x, box_y, box_w, box_h)
			pygame.draw.rect(self.game_surface, COLOR_BARS, box_rect)
			pygame.draw.rect(self.game_surface, (0, 0, 0), box_rect, 2)
			x = box_x + box_padding
			y = box_y + box_padding
			self.game_surface.blit(game_over_text, (x, y))
		if self.paused:
			title = self.font.render("SETTINGS", True, COLOR_GAME_OVER)
			title = pygame.transform.smoothscale(
				title,
				(
					max(1, title.get_width() // 2),
					max(1, title.get_height() // 2),
				),
			)
			lines = []
			for i in range(len(self.pause_options)):
				opt = self.pause_options[i]
				if opt == "MUSIC":
					lines.append(f"MUSIC: {int(self.music_volume * 100)}%")
				elif opt == "SFX":
					lines.append(f"SFX: {int(self.sfx_volume * 100)}%")
				elif opt == "CONTROL":
					lines.append(f"CONTROL: {self.control_mode}")
				elif opt == "INTENSITY":
					lines.append(f"INTENSITY: {cfg.INTENSITY.name}")
				else:
					lines.append(opt)
			rendered = []
			self.pause_option_rects = {}
			max_w = title.get_width()
			# Ensure width fits the longest INTENSITY label
			for i, line in enumerate(lines):
				if line.startswith("INTENSITY:"):
					line = "INTENSITY: MEDIUM"
				test = self.font.render("> " + line, True, COLOR_GAME_OVER)
				test = pygame.transform.smoothscale(
					test,
					(
						max(1, test.get_width() // 2),
						max(1, test.get_height() // 2),
					),
				)
				if test.get_width() > max_w:
					max_w = test.get_width()
			for i, line in enumerate(lines):
				selected = i == self.pause_index or i == self.hover_pause_index
				prefix = "> " if selected else "  "
				text = self.font.render(prefix + line, True, COLOR_GAME_OVER)
				text = pygame.transform.smoothscale(
					text,
					(
						max(1, text.get_width() // 2),
						max(1, text.get_height() // 2),
					),
				)
				rendered.append(text)
			total_h = title.get_height() + sum(t.get_height() for t in rendered) + 10 * len(rendered)
			box_padding = 12
			box_w = max_w + box_padding * 2
			box_h = total_h + box_padding * 2
			box_x = (WINDOW_WIDTH - box_w) // 2
			box_y = (WINDOW_HEIGHT - box_h) // 2 - 20
			box_rect = pygame.Rect(box_x, box_y, box_w, box_h)
			pygame.draw.rect(self.game_surface, COLOR_BARS, box_rect)
			pygame.draw.rect(self.game_surface, (0, 0, 0), box_rect, 2)
			x = box_x + box_padding
			y = box_y + box_padding
			self.game_surface.blit(title, (x, y))
			y += title.get_height() + 10
			for i, t in enumerate(rendered):
				self.game_surface.blit(t, (x, y))
				self.pause_option_rects[self.pause_options[i]] = pygame.Rect(
					x, y, t.get_width(), t.get_height()
				)
				y += t.get_height() + 10

	def draw(self):
		"""Render the game state."""
		screen_w, screen_h = self.screen.get_size()
		if self.in_menu:
			self._draw_menu_screen(screen_w, screen_h)
		elif self.in_health:
			self._draw_health_screen(screen_w, screen_h)
		else:
			self.draw_background()

			# Draw platforms and player
			for platform in self.platforms:
				platform.draw(self.game_surface)
			for coin in self.coins:
				coin.draw(self.game_surface)

			# Draw tutorial elements (portal + hint text)
			if self.in_tutorial:
				self._draw_portal()
				hint = self.subtitle_font.render(
					"JUMP TO THE PORTAL!", True, COLOR_TEXT
				)
				hint_x = (WINDOW_WIDTH - hint.get_width()) // 2
				self.game_surface.blit(hint, (hint_x, WINDOW_HEIGHT - 60))

			self.player.draw(self.game_surface)

			# Draw UI
			self.draw_ui()
			self._draw_countdown()

			scale = min(screen_w / WINDOW_WIDTH, screen_h / WINDOW_HEIGHT)
			scaled_w = int(WINDOW_WIDTH * scale)
			scaled_h = int(WINDOW_HEIGHT * scale)
			offset_x = (screen_w - scaled_w) // 2
			offset_y = (screen_h - scaled_h) // 2
			self.render_scale = scale
			self.render_offset = (offset_x, offset_y)

			self.screen.fill(COLOR_BARS)
			scaled_surface = pygame.transform.smoothscale(self.game_surface, (scaled_w, scaled_h))
			self.screen.blit(scaled_surface, (offset_x, offset_y))
			if offset_x > 0:
				bar_w = offset_x
				if self.bar_pattern_size != (bar_w, screen_h):
					self.bar_pattern_surface = self._build_bar_pattern(bar_w, screen_h)
					self.bar_pattern_size = (bar_w, screen_h)
				self.screen.blit(self.bar_pattern_surface, (0, 0))
				self.screen.blit(self.bar_pattern_surface, (offset_x + scaled_w, 0))
				left_border = pygame.Rect(offset_x - 2, 0, 2, screen_h)
				right_border = pygame.Rect(offset_x + scaled_w, 0, 2, screen_h)
				pygame.draw.rect(self.screen, (0, 0, 0), left_border)
				pygame.draw.rect(self.screen, (0, 0, 0), right_border)

			if offset_x > 0 and not self.in_tutorial:
				text_lines = [
					f"HIGHSCORE: {self.high_score}",
					f"SCORE: {self.score}",
					f"HEIGHT: {self.height_jumped:.1f}M",
					f"COINS: {self.coins_collected}",
				]
				tx = 16
				ty = 20
				for line in text_lines:
					text_surf = self.ui_font.render(line, True, COLOR_BARS_TEXT)
					self.screen.blit(text_surf, (tx, ty))
					ty += text_surf.get_height() + 10

				cal_text = self.ui_font.render(
					f"CALORIES: {self.display_calories:.2f}", True, COLOR_BARS_TEXT
				)
				right_bar_left = offset_x + scaled_w
				cal_x = right_bar_left + 16
				self.screen.blit(cal_text, (cal_x, 20))

			self._draw_cv_overlay(screen_w, screen_h)

		pygame.display.flip()
		self._update_cursor()

	def _draw_cv_overlay(self, screen_w, screen_h):
		"""Draw CV controls and debug feed in bottom-left corner."""
		if not self.cv_state:
			return

		controls_w = 180
		controls_h = 105
		padding = 24
		debug_gap = 20
		debug_target_w = 240
		x = padding
		y = screen_h - padding - controls_h

		self._draw_controls_widget(x, y, controls_w, controls_h)
		debug_h = self._estimate_debug_height(debug_target_w)
		debug_top = y - debug_gap - debug_h
		if self.jump_flash_timer > 0:
			label = self.subtitle_font.render("JUMP", True, COLOR_BARS_TEXT)
			self.screen.blit(label, (x, debug_top - label.get_height() - 6))
		self._draw_debug_frame(x, debug_top, debug_target_w)

	def _draw_controls_widget(self, x, y, w, h):
		active_color = (*COLOR_BARS_TEXT, 220)
		surf = pygame.Surface((w, h), pygame.SRCALPHA)

		center = (w // 2, h // 2)
		arrow_size = 32

		vec = None
		if self.cv_state.jump_active and self.cv_state.jump_vector is not None:
			vec = self.cv_state.jump_vector
		elif self.cv_state.center_velocity is not None:
			vx, vy = self.cv_state.center_velocity
			if math.hypot(vx, vy) >= CV_MOVE_DEADZONE:
				vec = self.cv_state.center_velocity

		self._draw_direction_arrow(
			surf,
			center,
			arrow_size,
			vec,
			active_color,
		)
		
		self.screen.blit(surf, (x, y))

	def _draw_direction_arrow(self, surf, center, size, vector, color):
		cx, cy = center
		if vector is None:
			vx, vy = 0.0, -1.0
		else:
			vx, vy = vector
			if abs(vx) < 1e-3 and abs(vy) < 1e-3:
				vx, vy = 0.0, -1.0
		angle = math.atan2(vy, vx) + math.pi / 2.0
		cos_a = math.cos(angle)
		sin_a = math.sin(angle)

		head = (0, -size)
		left = (-size * 0.5, size * 0.6)
		right = (size * 0.5, size * 0.6)

		def rot(pt):
			px, py = pt
			return (cx + px * cos_a - py * sin_a, cy + px * sin_a + py * cos_a)

		points = [rot(head), rot(left), rot(right)]
		pygame.draw.polygon(surf, color, points, 0)

	def _draw_debug_frame(self, x, y, target_w):
		if not self.cv_state or self.cv_state.debug_frame is None:
			return
		frame = self.cv_state.debug_frame
		if frame is None:
			return
		h, w = frame.shape[:2]
		if w == 0 or h == 0:
			return
		target_h = max(1, int(h * (target_w / w)))
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
		surf = pygame.transform.smoothscale(surf, (target_w, target_h))
		self.screen.blit(surf, (x, y))

	def _estimate_debug_height(self, target_w):
		if not self.cv_state or self.cv_state.debug_frame is None:
			return int(target_w * 0.75)
		frame = self.cv_state.debug_frame
		h, w = frame.shape[:2]
		if w == 0 or h == 0:
			return int(target_w * 0.75)
		return max(1, int(h * (target_w / w)))

	def _draw_countdown(self):
		if not self.countdown_active:
			return
		if self.countdown_remaining <= 0:
			return
		value = max(1, int(math.ceil(self.countdown_remaining)))
		text = self.title_font.render(str(value), True, COLOR_BARS_TEXT)
		scale = 2.5
		text = pygame.transform.smoothscale(
			text,
			(
				max(1, int(text.get_width() * scale)),
				max(1, int(text.get_height() * scale)),
			),
		)
		x = (WINDOW_WIDTH - text.get_width()) // 2
		y = (WINDOW_HEIGHT - text.get_height()) // 2
		self.game_surface.blit(text, (x, y))

	def _update_cursor(self):
		"""Update mouse cursor based on hover state."""
		hover = False
		self.hover_menu_index = None
		self.hover_pause_index = None
		mx, my = pygame.mouse.get_pos()
		if self.in_menu:
			for i, key in enumerate(self.menu_options):
				rect = self.menu_option_rects.get(key)
				if rect and rect.collidepoint(mx, my):
					hover = True
					self.hover_menu_index = i
					break
		elif self.paused:
			game_pos = self._screen_to_game((mx, my))
			if game_pos:
				gx, gy = game_pos
				for i, key in enumerate(self.pause_options):
					rect = self.pause_option_rects.get(key)
					if rect and rect.collidepoint(gx, gy):
						hover = True
						self.hover_pause_index = i
						break
		if self.hover_menu_index is not None:
			self.menu_index = self.hover_menu_index
		if self.hover_pause_index is not None:
			self.pause_index = self.hover_pause_index
		cursor = pygame.SYSTEM_CURSOR_HAND if hover else pygame.SYSTEM_CURSOR_ARROW
		pygame.mouse.set_cursor(cursor)

	def _draw_menu_screen(self, screen_w, screen_h):
		if self.menu_pattern_size != (screen_w, screen_h):
			self.menu_pattern_surface = self._build_menu_pattern(screen_w, screen_h)
			self.menu_pattern_size = (screen_w, screen_h)
		self.screen.blit(self.menu_pattern_surface, (0, 0))

		step = self.title_font.render("STEP", True, COLOR_BARS_TEXT)
		up = self.title_font.render("UP!", True, COLOR_BARS_TEXT)
		subtitle = self.subtitle_font.render("THE HEALTH  PLATFORMER", True, COLOR_BARS_TEXT)

		title_w = step.get_width() + up.get_width()
		title_x = (screen_w - title_w) // 2
		block_h = step.get_height() + 10 + subtitle.get_height() + 40 + self.menu_font.get_height() + 8
		title_y = (screen_h - block_h) // 2
		self.screen.blit(step, (title_x, title_y))
		up_offset = max(1, int(step.get_height() * 0.25))
		self.screen.blit(up, (title_x + step.get_width(), title_y - up_offset))

		sub_x = (screen_w - subtitle.get_width()) // 2
		sub_y = title_y + step.get_height() + 10
		self.screen.blit(subtitle, (sub_x, sub_y))

		prefix_w = self.subtitle_font.size("THE HEALTH ")[0]
		platform_w = self.subtitle_font.size("PLATFORM")[0]
		underline_y = sub_y + subtitle.get_height() - 2
		underline_x = sub_x + prefix_w
		pygame.draw.line(
			self.screen,
			COLOR_BARS_TEXT,
			(underline_x, underline_y),
			(underline_x + platform_w, underline_y),
			6,
		)

		self.menu_option_rects = {}
		options = []
		for i, opt in enumerate(self.menu_options):
			selected = i == self.menu_index or i == self.hover_menu_index
			prefix = "> " if selected else "  "
			options.append(self.menu_font.render(prefix + opt, True, COLOR_BARS_TEXT))
		spacing = 30
		total_w = sum(o.get_width() for o in options) + spacing * (len(options) - 1)
		start_x = (screen_w - total_w) // 2
		ty = sub_y + subtitle.get_height() + 40
		x = start_x
		self.menu_option_rects = {}
		for i, text in enumerate(options):
			self.screen.blit(text, (x, ty))
			self.menu_option_rects[self.menu_options[i]] = pygame.Rect(
				x, ty, text.get_width(), text.get_height()
			)
			x += text.get_width() + spacing

		# Sign-in prompt / status
		if self.signin_mode:
			label = f"USERNAME: {self.username_input}_"
			prompt = self.subtitle_font.render(label, True, COLOR_BARS_TEXT)
			self.screen.blit(
				prompt,
				((screen_w - prompt.get_width()) // 2, ty + text.get_height() + 16),
			)
		elif self.current_user:
			status = self.subtitle_font.render(
				f"SIGNED IN: {self.current_user}", True, COLOR_BARS_TEXT
			)
			self.screen.blit(
				status,
				((screen_w - status.get_width()) // 2, ty + text.get_height() + 16),
			)
		self.signin_rect = None

	def _draw_health_screen(self, screen_w, screen_h):
		if self.menu_pattern_size != (screen_w, screen_h):
			self.menu_pattern_surface = self._build_menu_pattern(screen_w, screen_h)
			self.menu_pattern_size = (screen_w, screen_h)
		self.screen.blit(self.menu_pattern_surface, (0, 0))
		title = self.subtitle_font.render("HEALTH INFO", True, COLOR_BARS_TEXT)
		if not self.current_user:
			sub = self.subtitle_font.render("SIGN IN TO VIEW METRICS", True, COLOR_BARS_TEXT)
			self.screen.blit(sub, ((screen_w - sub.get_width()) // 2, screen_h // 2 - 20))
			return
		user = self.users.get(self.current_user, {})
		lines = [
			f"USER: {self.current_user}",
			f"HIGHSCORE: {user.get('highscore', 0)}",
			f"LIFETIME CALORIES: {user.get('lifetime_calories', 0.0):.2f}",
			f"MINUTES PLAYED: {user.get('minutes_played', 0.0):.1f}",
			f"BALANCE ABILITY: {user.get('balance_ability', 0.0):.2f}",
		]
		line_surfs = [self.subtitle_font.render(line, True, COLOR_BARS_TEXT) for line in lines]
		max_w = max(s.get_width() for s in line_surfs)
		total_h = sum(s.get_height() for s in line_surfs) + 8 * (len(line_surfs) - 1)
		sprite = Player.sprite_stand if Player.sprites_loaded else None
		gap = 20
		block_w = max_w
		sprite_big = None
		if sprite:
			scale = 6
			sprite_big = pygame.transform.smoothscale(
				sprite, (sprite.get_width() * scale, sprite.get_height() * scale)
			)
			block_w = max_w + sprite_big.get_width() + gap

		block_h = total_h
		if sprite_big:
			block_h = max(total_h, sprite_big.get_height())

		start_x = (screen_w - block_w) // 2
		start_y = (screen_h - block_h) // 2 + 20

		title_x = start_x + (block_w - title.get_width()) // 2
		title_y = start_y - title.get_height() - 16
		self.screen.blit(title, (title_x, title_y))

		if sprite_big:
			sprite_x = start_x + max_w + gap
			sprite_y = start_y + (block_h - sprite_big.get_height()) // 2
			self.screen.blit(sprite_big, (sprite_x, sprite_y))
			text_x = start_x
			text_y = start_y + (block_h - total_h) // 2
		else:
			text_x = start_x
			text_y = start_y

		y = text_y
		for s in line_surfs:
			self.screen.blit(s, (text_x, y))
			y += s.get_height() + 8

	def run(self):
		"""Main game loop."""
		while self.running:
			dt = self.clock.tick(FPS) / 1000.0

			self.handle_events()
			self.update(dt)
			self.draw()

		if self.cv:
			self.cv.stop()
		pygame.quit()
		sys.exit()

