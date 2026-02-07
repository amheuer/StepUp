"""Main game loop and rendering."""

import sys
import random
from pathlib import Path

import pygame

from collisions import check_if_on_platform
from coins import Coin, collect_coins, cull_coins, spawn_coins_near_platforms
from config import (
	COLOR_BARS,
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
	INTENSITY,
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
		self.game_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
		self.high_score = 0
		self.next_high_score_sfx = 1000
		tileset = pygame.image.load(TILESET_PATH).convert_alpha()
		Platform.load_tileset(tileset)
		self._load_coin_frames()
		self.sfx = self._load_sfx()
		pygame.mixer.music.load(SOUNDTRACK_PATH)
		pygame.mixer.music.play(-1)
		self.skip_sfx_frames = 0
		self._load_backgrounds()
		self.particle_surface_far = self._build_particle_layer(50)
		self.particle_surface_near = self._build_particle_layer(90)

		self.reset()
		self.running = True
		self.game_over = False

	def reset(self):
		"""Reset game state for a new game."""
		self.player = Player(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 80)
		self.platforms = create_initial_platforms()
		self.coins = spawn_coins_near_platforms(self.platforms)
		self.score = 0
		self.height_jumped = 0
		self.coins_collected = 0
		self.calories = 0.0
		self.display_calories = 0.0
		self.next_high_score_sfx = 1000
		pygame.mixer.music.play(-1)
		self.skip_sfx_frames = 2

	def _update_calories(self):
		"""Return calories per minute based on intensity."""
		return INTENSITY.value

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

	def handle_events(self):
		"""Process input events."""
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					self.running = False
				elif self.game_over and event.key == pygame.K_r:
					self.reset()
					self.game_over = False

	def update(self, dt):
		"""Update game state."""
		if self.game_over:
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
			self.score += int(dy)
			if self.score > self.height_jumped:
				self.height_jumped = self.score
			if self.score > self.high_score:
				self.high_score = self.score

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
		self.score += coin_score
		self.coins_collected += coin_count
		if play_sfx and coin_count > 0:
			self.sfx["coin.wav"].play()
		if self.score > self.high_score:
			self.high_score = self.score
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

	def draw_background(self):
		"""Draw layered background with parallax bubbles."""
		base = pygame.transform.scale(self.bg_base, (WINDOW_WIDTH, WINDOW_HEIGHT))
		self.game_surface.blit(base, (0, 0))

		self._blit_vertical_tiled(self.particle_surface_far, int(-self.score * 0.03))
		self._blit_vertical_tiled(self.particle_surface_near, int(-self.score * 0.06))

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

	def draw(self):
		"""Render the game state."""
		self.draw_background()

		# Draw platforms and player
		for platform in self.platforms:
			platform.draw(self.game_surface)
		for coin in self.coins:
			coin.draw(self.game_surface)
		self.player.draw(self.game_surface)

		# Draw UI
		self.draw_ui()

		screen_w, screen_h = self.screen.get_size()
		scale = min(screen_w / WINDOW_WIDTH, screen_h / WINDOW_HEIGHT)
		scaled_w = int(WINDOW_WIDTH * scale)
		scaled_h = int(WINDOW_HEIGHT * scale)
		offset_x = (screen_w - scaled_w) // 2
		offset_y = (screen_h - scaled_h) // 2

		self.screen.fill(COLOR_BARS)
		scaled_surface = pygame.transform.smoothscale(self.game_surface, (scaled_w, scaled_h))
		self.screen.blit(scaled_surface, (offset_x, offset_y))
		if offset_x > 0:
			left_border = pygame.Rect(offset_x - 2, 0, 2, screen_h)
			right_border = pygame.Rect(offset_x + scaled_w, 0, 2, screen_h)
			pygame.draw.rect(self.screen, (0, 0, 0), left_border)
			pygame.draw.rect(self.screen, (0, 0, 0), right_border)

		if offset_x > 0:
			text_lines = [
				f"HIGHSCORE: {self.high_score}",
				f"SCORE: {self.score}",
				f"HEIGHT: {self.height_jumped}",
				f"COINS: {self.coins_collected}",
			]
			tx = 16
			ty = 20
			for line in text_lines:
				text_surf = self.ui_font.render(line, True, COLOR_BARS_TEXT)
				self.screen.blit(text_surf, (tx, ty))
				ty += text_surf.get_height() + 10

			cal_text = self.ui_font.render(
				f"CALORIES: {self.display_calories:.1f}", True, COLOR_BARS_TEXT
			)
			right_bar_left = offset_x + scaled_w
			cal_x = right_bar_left + 16
			self.screen.blit(cal_text, (cal_x, 20))

		pygame.display.flip()

	def run(self):
		"""Main game loop."""
		while self.running:
			dt = self.clock.tick(FPS) / 1000.0

			self.handle_events()
			self.update(dt)
			self.draw()

		pygame.quit()
		sys.exit()
