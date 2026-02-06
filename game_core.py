"""Main game loop and rendering."""

import sys

import pygame

from collisions import check_if_on_platform
from config import (
	COLOR_BG,
	COLOR_GAME_OVER,
	COLOR_STAR,
	COLOR_TEXT,
	FPS,
	SCROLL_THRESHOLD,
	WINDOW_HEIGHT,
	WINDOW_TITLE,
	WINDOW_WIDTH,
)
from entities import Player
from platforms import create_initial_platforms, generate_new_platforms


class Game:
	"""Main game manager."""

	def __init__(self):
		pygame.init()
		self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
		pygame.display.set_caption(WINDOW_TITLE)
		self.clock = pygame.time.Clock()
		self.font = pygame.font.SysFont(None, 32)

		self.reset()
		self.running = True
		self.game_over = False

	def reset(self):
		"""Reset game state for a new game."""
		self.player = Player(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 80)
		self.platforms = create_initial_platforms()
		self.score = 0

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
			platform_below.land()  # Start fading this platform

		# Handle scrolling
		if self.player.y < SCROLL_THRESHOLD:
			dy = SCROLL_THRESHOLD - self.player.y
			self.player.y = SCROLL_THRESHOLD
			for platform in self.platforms:
				platform.y += dy
			self.score += int(dy)

		# Remove off-screen or faded platforms
		self.platforms = [
			p for p in self.platforms if p.y < WINDOW_HEIGHT + 50 and p.is_active
		]

		# Generate new platforms
		generate_new_platforms(self.platforms)

		# Check game over condition
		if self.player.y - self.player.radius > WINDOW_HEIGHT:
			self.game_over = True

	def draw_background(self):
		"""Draw background with parallax stars."""
		self.screen.fill(COLOR_BG)

		for i in range(30):
			sx = (i * 37 + self.score) % WINDOW_WIDTH
			sy = (i * 53 + self.score // 3) % WINDOW_HEIGHT
			self.screen.set_at((sx, sy), COLOR_STAR)

	def draw_ui(self):
		"""Draw score and game over text."""
		score_text = self.font.render(f"Score: {self.score}", True, COLOR_TEXT)
		self.screen.blit(score_text, (8, 8))

		if self.game_over:
			game_over_text = self.font.render(
				"Game Over - Press R to restart",
				True,
				COLOR_GAME_OVER,
			)
			x = (WINDOW_WIDTH - game_over_text.get_width()) // 2
			y = WINDOW_HEIGHT // 2 - 20
			self.screen.blit(game_over_text, (x, y))

	def draw(self):
		"""Render the game state."""
		self.draw_background()

		# Draw platforms and player
		for platform in self.platforms:
			platform.draw(self.screen)
		self.player.draw(self.screen)

		# Draw UI
		self.draw_ui()

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
