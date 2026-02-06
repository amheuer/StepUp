"""
StepUp: A minimal Doodle Jump clone using Pygame.
- Player is a circle
- Platforms are rectangles
- No external assets required
"""

import pygame
import random
import sys


# ============================================================================
# CONSTANTS
# ============================================================================

# Window
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 600
WINDOW_TITLE = "StepUp"
FPS = 60

# Physics
GRAVITY = 800
JUMP_STRENGTH = 420
PLAYER_MOVE_SPEED = 200
PLAYER_RADIUS = 16

# Platform
PLATFORM_HEIGHT = 12
PLATFORM_BASE_WIDTH = 60
MOVING_PLATFORM_CHANCE = 0.12
MOVING_PLATFORM_SPEED_MIN = 30
MOVING_PLATFORM_SPEED_MAX = 70

# Scroll
SCROLL_THRESHOLD = WINDOW_HEIGHT * 0.4

# Rendering
COLOR_BG = (15, 18, 30)
COLOR_PLAYER = (30, 120, 200)
COLOR_PLATFORM = (50, 200, 110)
COLOR_STAR = (200, 200, 220)
COLOR_TEXT = (230, 230, 230)
COLOR_TEXT_SECONDARY = (200, 200, 200)
COLOR_GAME_OVER = (255, 200, 200)


# ============================================================================
# PLAYER CLASS
# ============================================================================

class Player:
	"""Represents the jumping player character."""
	
	def __init__(self, x, y, radius=PLAYER_RADIUS):
		self.x = x
		self.y = y
		self.radius = radius
		self.vx = 0
		self.vy = 0
		self.color = COLOR_PLAYER
	
	@property
	def rect(self):
		"""Bounding rectangle for collision detection."""
		return pygame.Rect(
			self.x - self.radius,
			self.y - self.radius,
			self.radius * 2,
			self.radius * 2
		)
	
	def update(self, dt, keys):
		"""Update player position and velocity."""
		# Horizontal movement based on input
		ax = 0
		if keys[pygame.K_LEFT] or keys[pygame.K_a]:
			ax -= 1
		if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
			ax += 1
		self.vx = ax * PLAYER_MOVE_SPEED
		
		# Update position
		self.x += self.vx * dt
		self.y += self.vy * dt
		
		# Apply gravity
		self.vy += GRAVITY * dt
		
		# Wrap horizontally (screen edges)
		if self.x < -self.radius:
			self.x = WINDOW_WIDTH + self.radius
		elif self.x > WINDOW_WIDTH + self.radius:
			self.x = -self.radius
	
	def jump(self, strength=JUMP_STRENGTH):
		"""Apply a jump impulse."""
		self.vy = -strength
	
	def draw(self, surface):
		"""Draw the player as a circle."""
		pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)


# ============================================================================
# PLATFORM CLASS
# ============================================================================

class Platform:
	"""Represents a platform the player can jump on."""
	
	def __init__(self, x, y, width=PLATFORM_BASE_WIDTH, height=PLATFORM_HEIGHT, moving=False):
		self.x = x
		self.y = y
		self.w = width
		self.h = height
		self.color = COLOR_PLATFORM
		self.moving = moving
		self.dir = 1 if random.random() < 0.5 else -1
		self.speed = random.uniform(MOVING_PLATFORM_SPEED_MIN, MOVING_PLATFORM_SPEED_MAX) if moving else 0
	
	@property
	def rect(self):
		"""Bounding rectangle for collision detection."""
		return pygame.Rect(self.x, self.y, self.w, self.h)
	
	def update(self, dt):
		"""Update platform position if moving."""
		if self.moving:
			self.x += self.dir * self.speed * dt
			# Bounce off screen edges
			if self.x < 0:
				self.x = 0
				self.dir *= -1
			elif self.x + self.w > WINDOW_WIDTH:
				self.x = WINDOW_WIDTH - self.w
				self.dir *= -1
	
	def draw(self, surface):
		"""Draw the platform as a rounded rectangle."""
		pygame.draw.rect(surface, self.color, self.rect, border_radius=4)


# ============================================================================
# COLLISION DETECTION
# ============================================================================

def check_platform_collision(player, platforms, prev_x, prev_y):
	"""
	Detect collision between player and platforms.
	Returns the platform collided with, or None.
	Only collides when player is falling downward.
	"""
	if player.vy <= 0:
		return None
	
	closest_platform = None
	closest_distance = float('inf')
	
	# Create a rect for the player's previous position
	prev_rect = pygame.Rect(
		prev_x - player.radius,
		prev_y - player.radius,
		player.radius * 2,
		player.radius * 2
	)
	
	for platform in platforms:
		# Detect fresh collision: wasn't colliding before, but is now
		if not prev_rect.colliderect(platform.rect) and player.rect.colliderect(platform.rect):
			# Find the closest platform (in case of overlap)
			distance = player.y + player.radius - platform.y
			if distance < closest_distance:
				closest_distance = distance
				closest_platform = platform
	
	return closest_platform


# ============================================================================
# PLATFORM GENERATION
# ============================================================================

def create_initial_platforms():
	"""Generate the initial set of platforms."""
	platforms = []
	
	# Base platform at the bottom
	platforms.append(Platform(
		WINDOW_WIDTH // 2 - 40,
		WINDOW_HEIGHT - 40,
		width=80,
		height=14
	))
	
	# Generate platforms going upward
	y = WINDOW_HEIGHT - 120
	while y > -2000:
		x = random.randint(0, WINDOW_WIDTH - 60)
		is_moving = random.random() < MOVING_PLATFORM_CHANCE
		platforms.append(Platform(
			x, y,
			width=random.randint(50, 80),
			height=12,
			moving=is_moving
		))
		y -= random.randint(60, 120)
		if len(platforms) > 40:
			break
	
	return platforms


def generate_new_platforms(platforms):
	"""Generate new platforms above existing ones."""
	while len(platforms) < 10:
		top_y = min((p.y for p in platforms), default=0)
		new_y = top_y - random.randint(60, 140)
		new_x = random.randint(0, WINDOW_WIDTH - 60)
		is_moving = random.random() < MOVING_PLATFORM_CHANCE
		platforms.append(Platform(
			new_x, new_y,
			width=random.randint(50, 90),
			moving=is_moving
		))


# ============================================================================
# GAME CLASS
# ============================================================================

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
		prev_x = self.player.x
		prev_y = self.player.y
		
		# Update entities
		self.player.update(dt, keys)
		for platform in self.platforms:
			platform.update(dt)
		
		# Handle collision
		collided_platform = check_platform_collision(self.player, self.platforms, prev_x, prev_y)
		if collided_platform:
			self.player.y = collided_platform.y - self.player.radius
			self.player.jump()
		
		# Handle scrolling
		if self.player.y < SCROLL_THRESHOLD:
			dy = SCROLL_THRESHOLD - self.player.y
			self.player.y = SCROLL_THRESHOLD
			for platform in self.platforms:
				platform.y += dy
			self.score += int(dy)
		
		# Remove off-screen platforms
		self.platforms = [p for p in self.platforms if p.y < WINDOW_HEIGHT + 50]
		
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
			game_over_text = self.font.render("Game Over - Press R to restart", True, COLOR_GAME_OVER)
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


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
	game = Game()
	game.run()
