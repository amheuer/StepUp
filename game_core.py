"""Main game loop and rendering."""

import sys
import random
from pathlib import Path

import pygame

from collisions import check_if_on_platform
from coins import collect_coins, cull_coins, spawn_coins_near_platforms
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
		self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
		pygame.display.set_caption(WINDOW_TITLE)
		self.clock = pygame.time.Clock()
		self.font = pygame.font.SysFont(None, 32)
		self.ui_font = pygame.font.SysFont(None, UI_FONT_SIZE)
		self.game_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
		self.high_score = 0
		tileset = pygame.image.load(TILESET_PATH).convert_alpha()
		Platform.load_tileset(tileset)
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

		# Update entities
		self.player.update(dt, keys)
		for platform in self.platforms:
			platform.update(dt)

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
		if self.score > self.high_score:
			self.high_score = self.score

		# Cull coins
		self.coins = cull_coins(self.coins)

		# Check game over condition
		if self.player.y - self.player.radius > WINDOW_HEIGHT:
			self.game_over = True

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
				"Game Over - Press R to restart",
				True,
				COLOR_GAME_OVER,
			)
			x = (WINDOW_WIDTH - game_over_text.get_width()) // 2
			y = WINDOW_HEIGHT // 2 - 20
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
			text_lines = [
				f"Highscore: {self.high_score}",
				f"Score: {self.score}",
				f"Height: {self.height_jumped}",
				f"Coins: {self.coins_collected}",
			]
			tx = 16
			ty = 20
			for line in text_lines:
				text_surf = self.ui_font.render(line, True, COLOR_BARS_TEXT)
				self.screen.blit(text_surf, (tx, ty))
				ty += text_surf.get_height() + 10

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
