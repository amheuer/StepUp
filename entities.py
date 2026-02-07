"""Game entities for StepUp."""

import random
import pygame

from config import (
	COLOR_PLAYER,
	COLOR_PLATFORM,
	COLOR_PLATFORM_FRAGILE,
	FRAGILE_PLATFORM_FADE_DURATION,
	GRAVITY,
	JUMP_STRENGTH,
	MOVING_PLATFORM_SPEED_MAX,
	MOVING_PLATFORM_SPEED_MIN,
	PLATFORM_FADE_DURATION,
	PLAYER_MOVE_SPEED,
	PLAYER_RADIUS,
	WINDOW_WIDTH,
)


class Player:
	"""Represents the jumping player character."""

	def __init__(self, x, y, radius=PLAYER_RADIUS):
		self.x = x
		self.y = y
		self.radius = radius
		self.vx = 0
		self.vy = 0
		self.color = COLOR_PLAYER
		self.can_jump = False  # Can jump when on a platform
		self.prev_up_pressed = False  # Track previous frame's UP key state
		self.on_platform = False  # Track if player is on a platform this frame

	@property
	def rect(self):
		"""Bounding rectangle for collision detection."""
		return pygame.Rect(
			self.x - self.radius,
			self.y - self.radius,
			self.radius * 2,
			self.radius * 2,
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

		# Jump on UP key press (not held)
		up_pressed = keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_SPACE]
		if up_pressed and not self.prev_up_pressed and self.can_jump:
			self.jump()
			self.can_jump = False
		self.prev_up_pressed = up_pressed

		# Update position
		self.x += self.vx * dt
		self.y += self.vy * dt

		# Apply gravity only when not on a platform or when already falling
		# (gravity applies while jumping/falling, not while standing on platform)
		if not self.on_platform or self.vy > 0:
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


class Platform:
	"""Represents a platform the player can jump on."""

	def __init__(self, x, y, width, height, kind="normal"):
		self.x = x
		self.y = y
		self.w = width
		self.h = height
		self.kind = kind
		self.color = COLOR_PLATFORM_FRAGILE if kind == "fragile" else COLOR_PLATFORM
		self.moving = kind == "moving"
		self.dir = 1 if random.random() < 0.5 else -1
		self.vx = 0
		self.speed = (
			random.uniform(MOVING_PLATFORM_SPEED_MIN, MOVING_PLATFORM_SPEED_MAX)
			if self.moving
			else 0
		)
		self.fade_duration = (
			FRAGILE_PLATFORM_FADE_DURATION if kind == "fragile" else PLATFORM_FADE_DURATION
		)

		# Lifetime tracking
		self.landed_time = None  # When player landed on this platform
		self.is_active = True  # Platform is still visible/active

	@property
	def rect(self):
		"""Bounding rectangle for collision detection."""
		return pygame.Rect(self.x, self.y, self.w, self.h)

	def land(self):
		"""Called when player lands on this platform. Only starts fading once."""
		if self.landed_time is None:
			self.landed_time = 0  # Start fading timer only on first landing

	def update(self, dt):
		"""Update platform position if moving, and fade if landed on."""
		self.vx = 0
		if self.moving:
			self.vx = self.dir * self.speed
			self.x += self.vx * dt
			# Bounce off screen edges
			if self.x < 0:
				self.x = 0
				self.dir *= -1
			elif self.x + self.w > WINDOW_WIDTH:
				self.x = WINDOW_WIDTH - self.w
				self.dir *= -1

		# Update fade timer
		if self.landed_time is not None:
			self.landed_time += dt
			if self.landed_time >= self.fade_duration:
				self.is_active = False

	def get_alpha(self):
		"""Return the alpha value (0-255) for drawing."""
		if self.landed_time is None:
			return 255

		# Fade from 255 to 0 over fade_duration seconds
		alpha = 255 * (1 - self.landed_time / self.fade_duration)
		return max(0, int(alpha))

	def draw(self, surface):
		"""Draw the platform as a rounded rectangle with fading."""
		if not self.is_active:
			return

		alpha = self.get_alpha()
		if alpha <= 0:
			return

		# Create a surface with per-pixel alpha for fading
		platform_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
		color_with_alpha = (*self.color, alpha)
		pygame.draw.rect(
			platform_surf,
			color_with_alpha,
			(0, 0, self.w, self.h),
			border_radius=4,
		)
		surface.blit(platform_surf, (self.x, self.y))
