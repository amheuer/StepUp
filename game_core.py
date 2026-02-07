"""Main game loop and rendering."""

import sys
import random
from pathlib import Path
import json

import pygame

from collisions import check_if_on_platform
from coins import Coin, collect_coins, cull_coins, spawn_coins_near_platforms
from config import (
	COLOR_BARS,
	COLOR_BARS_PATTERN,
	COLOR_BARS_TEXT,
	COLOR_GAME_OVER,
	COLOR_TEXT,
	FPS,
	SCROLL_THRESHOLD,
	UI_FONT_SIZE,
	WINDOW_HEIGHT,
	WINDOW_TITLE,
	WINDOW_WIDTH,
	COIN_RADIUS,
	COIN_VALUE,
	INTENSITY,
	PLAYER_RADIUS,
	PLAYER_HEIGHT_METERS,
)
from entities import Player, Platform
from platforms import create_initial_platforms, generate_new_platforms

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

		self.reset()
		self.running = True
		self.game_over = False
		self.paused = False
		self.pause_index = 0
		self.pause_options = ["RESUME", "MUSIC", "SFX", "RESTART", "MAIN MENU", "QUIT"]
		self.in_menu = True
		self.in_health = False
		self.menu_index = 0
		self.menu_options = ["PLAY", "HEALTH  INFO", "SIGN IN"]
		self.menu_option_rects = {}
		self.signin_rect = None
		self.pause_option_rects = {}
		self.render_scale = 1.0
		self.render_offset = (0, 0)
		self.signin_mode = False
		self.username_input = ""
		self.current_user = None
		self.users = self._load_users()
		self.user_save_timer = 0.0

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

	def _update_calories(self):
		"""Return calories per minute based on intensity."""
		return INTENSITY.value

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
			}
		self.current_user = key

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

	def _load_player_sprites(self):
		"""Load player sprites."""
		stand = pygame.image.load(PLAYER_DIR / "stand_still.png").convert_alpha()
		left = pygame.image.load(PLAYER_DIR / "move_left.png").convert_alpha()
		jump = pygame.image.load(PLAYER_DIR / "jump.png").convert_alpha()

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
								self.reset()
								self.in_menu = False
								self.in_health = False
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
							self.reset()
							self.in_menu = False
							self.in_health = False
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

		keys = pygame.key.get_pressed()

		# Store previous position for collision detection
		prev_y = self.player.y
		was_on_platform = self.player.on_platform
		prev_high_score = self.high_score
		play_sfx = self.skip_sfx_frames <= 0
		if self.skip_sfx_frames > 0:
			self.skip_sfx_frames -= 1

		# Update entities
		self.player.update(dt, keys)
		if play_sfx and self.player.jumped_this_frame:
			self.sfx["jump.wav"].play()

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
		platform_below = check_if_on_platform(self.player, self.platforms, prev_y)
		if platform_below:
			self.player.on_platform = True
			self.player.vy = 0  # Stop velocity when on platform (gravity won't apply while on platform)
			self.player.can_jump = True  # Allow player to jump
			self.player.x += platform_below.vx * dt
			platform_below.land()  # Start fading this platform
			if play_sfx and not was_on_platform:
				self.sfx["tap.wav"].play()
			if not was_on_platform:
				self.display_calories = self.calories

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
				else:
					lines.append(opt)
			rendered = []
			self.pause_option_rects = {}
			max_w = title.get_width()
			for i, line in enumerate(lines):
				prefix = "> " if i == self.pause_index else "  "
				text = self.font.render(prefix + line, True, COLOR_GAME_OVER)
				text = pygame.transform.smoothscale(
					text,
					(
						max(1, text.get_width() // 2),
						max(1, text.get_height() // 2),
					),
				)
				rendered.append(text)
				# Ensure width accounts for selector prefix even when not selected.
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
			self.player.draw(self.game_surface)

			# Draw UI
			self.draw_ui()

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

			if offset_x > 0:
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

		pygame.display.flip()
		self._update_cursor()

	def _update_cursor(self):
		"""Update mouse cursor based on hover state."""
		hover = False
		mx, my = pygame.mouse.get_pos()
		if self.in_menu:
			for rect in self.menu_option_rects.values():
				if rect.collidepoint(mx, my):
					hover = True
					break
		elif self.paused:
			game_pos = self._screen_to_game((mx, my))
			if game_pos:
				gx, gy = game_pos
				for rect in self.pause_option_rects.values():
					if rect.collidepoint(gx, gy):
						hover = True
						break
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
			prefix = "> " if i == self.menu_index else "  "
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

		pygame.quit()
		sys.exit()
