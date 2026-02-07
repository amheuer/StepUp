"""Game entities for StepUp."""

import math
import random
import pygame

from config import (
	COLOR_PLAYER,
	COLOR_PLATFORM,
	COLOR_PLATFORM_FRAGILE,
	get_fragile_fade_duration,
	GRAVITY,
	JUMP_COOLDOWN,
	JUMP_REARM_TIME,
	JUMP_STRENGTH,
	get_platform_fade_duration,
	MOVING_PLATFORM_SPEED_MAX,
	MOVING_PLATFORM_SPEED_MIN,
	PLATFORM_FADE_DURATION,
	PLAYER_MAX_X_SPEED,
	PLAYER_MAX_SPEED,
	PLAYER_MOVE_SPEED,
	PLAYER_RADIUS,
	WINDOW_WIDTH,
)


class Player:
	"""Represents the jumping player character."""

	sprites_loaded = False
	sprite_stand = None
	sprite_left = None
	sprite_jump = None

	@classmethod
	def load_sprites(cls, stand, left, right, jump):
		cls.sprite_stand = stand
		cls.sprite_left = left
		cls.sprite_right = right
		cls.sprite_jump = jump
		cls.sprites_loaded = True

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
		self.jumped_this_frame = False
		self.jump_cooldown = 0.0
		self.ignore_platform_timer = 0.0
		self.time_since_jump = None


	@property
	def rect(self):
		"""Bounding rectangle for collision detection."""
		return pygame.Rect(
			self.x - self.radius,
			self.y - self.radius,
			self.radius * 2,
			self.radius * 2,
		)

	def update(self, dt, move_vx, jump_pressed, jump_velocity=None):
		"""Update player position and velocity."""
		self.jumped_this_frame = False
		if self.jump_cooldown > 0:
			self.jump_cooldown = max(0.0, self.jump_cooldown - dt)
		if self.time_since_jump is not None:
			self.time_since_jump += dt

		# Horizontal movement based on input only while grounded.
		if self.on_platform:
			if move_vx is None:
				self.vx = 0
			else:
				self.vx = max(-PLAYER_MAX_X_SPEED, min(PLAYER_MAX_X_SPEED, move_vx))

		# Jump on UP key press (not held), respecting cooldown and upward lock.
		if (
			jump_pressed
			and not self.prev_up_pressed
			and self.can_jump
			and self.jump_cooldown <= 0.0
			and self.vy >= 0
			and (self.time_since_jump is None or self.time_since_jump >= JUMP_REARM_TIME)
		):
			self.jump(jump_velocity)
			self.can_jump = False
			self.jumped_this_frame = True
			self.jump_cooldown = JUMP_COOLDOWN
			self.on_platform = False
			self.time_since_jump = 0.0
		self.prev_up_pressed = jump_pressed

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

	def jump(self, velocity=None, strength=JUMP_STRENGTH):
		"""Apply a jump impulse."""
		if velocity is None:
			self.vx = 0
			self.vy = -strength
			return
		vx, vy = velocity
		if vx is None or vy is None:
			self.vx = 0
			self.vy = -strength
			return
		speed = math.hypot(vx, vy)
		if speed <= 1e-6:
			self.vx = 0
			self.vy = -strength
			return
		nx = vx / speed
		ny = vy / speed
		raw_vx = nx * strength
		clamped_vx = max(-PLAYER_MAX_X_SPEED, min(PLAYER_MAX_X_SPEED, raw_vx))
		remaining = max(0.0, (strength * strength) - (clamped_vx * clamped_vx))
		vy_mag = math.sqrt(remaining)
		self.vx = clamped_vx
		self.vy = -vy_mag if ny < 0 else vy_mag

	def draw(self, surface):
		"""Draw the player as a circle or sprite."""
		if Player.sprites_loaded:
			if self.vy < -1:
				sprite = Player.sprite_jump
			elif self.vx < -1:
				sprite = Player.sprite_left
			elif self.vx > 1:
				sprite = Player.sprite_right
			else:
				sprite = Player.sprite_stand
			rect = sprite.get_rect(midbottom=(int(self.x), int(self.y + self.radius)))
			surface.blit(sprite, rect)
			return
		pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)


class Platform:
	"""Represents a platform the player can jump on."""

	tileset_loaded = False
	tiles_normal = None
	tiles_fragile = None
	tiles_scaled_cache = {}

	@classmethod
	def load_tileset(cls, tileset_surface):
		"""Load platform tiles from the tileset surface."""
		tileset_w = tileset_surface.get_width()
		tile_w = tileset_w // 8
		tile_h = tile_w

		def tile_at(row, col):
			rect = pygame.Rect(col * tile_w, row * tile_h, tile_w, tile_h)
			return tileset_surface.subsurface(rect).copy()

		cls.tiles_normal = (
			tile_at(3, 0),
			tile_at(3, 1),
			tile_at(3, 2),
		)
		cls.tiles_fragile = (
			tile_at(4, 0),
			tile_at(4, 1),
			tile_at(4, 2),
		)
		cls.tileset_loaded = True
		cls.tiles_scaled_cache.clear()

	@classmethod
	def _get_scaled_tiles(cls, kind, height):
		key = (kind, height)
		if key in cls.tiles_scaled_cache:
			return cls.tiles_scaled_cache[key]
		if kind == "fragile":
			left, mid, right = cls.tiles_fragile
		else:
			left, mid, right = cls.tiles_normal
		tile_w = height
		left_s = pygame.transform.scale(left, (tile_w, height))
		mid_s = pygame.transform.scale(mid, (tile_w, height))
		right_s = pygame.transform.scale(right, (tile_w, height))
		cls.tiles_scaled_cache[key] = (left_s, mid_s, right_s)
		return left_s, mid_s, right_s

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
			get_fragile_fade_duration() if kind == "fragile" else get_platform_fade_duration()
		)

		# Lifetime tracking
		self.landed_time = None  # When player landed on this platform
		self.is_active = True  # Platform is still visible/active
		self.crack_seed = random.randint(0, 1_000_000)
		self.crack_lines_max = 0
		self.crack_alpha_max = 0
		self.just_broke = False

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
		self.just_broke = False
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
				if self.is_active:
					self.just_broke = True
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

		crack_progress = 1 - (alpha / 255)
		crack_alpha = max(0, int(alpha * 0.7))
		crack_lines = 0
		if crack_progress > 0.15:
			crack_lines = 2
		if crack_progress > 0.4:
			crack_lines = 4
		if crack_progress > 0.7:
			crack_lines = 6
		self.crack_lines_max = max(self.crack_lines_max, crack_lines)
		self.crack_alpha_max = max(self.crack_alpha_max, crack_alpha)

		if Platform.tileset_loaded:
			left, mid, right = Platform._get_scaled_tiles(self.kind, self.h)

			tile_w = left.get_width()
			draw_x = int(self.x)
			draw_y = int(self.y)
			right_x = self.w - tile_w

			# Build a mask surface from full-alpha tiles to constrain cracks.
			mask_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
			x = 0
			if self.w <= tile_w * 2:
				mask_surf.blit(left, (x, 0))
				mask_surf.blit(right, (right_x, 0))
			else:
				mask_surf.blit(left, (x, 0))
				x += tile_w
				while x <= right_x - tile_w:
					mask_surf.blit(mid, (x, 0))
					x += tile_w
				mask_surf.blit(right, (right_x, 0))

			# Build the visible platform surface with fade alpha.
			left_d = left.copy()
			mid_d = mid.copy()
			right_d = right.copy()
			left_d.set_alpha(alpha)
			mid_d.set_alpha(alpha)
			right_d.set_alpha(alpha)

			platform_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
			x = 0
			if self.w <= tile_w * 2:
				platform_surf.blit(left_d, (x, 0))
				platform_surf.blit(right_d, (right_x, 0))
			else:
				platform_surf.blit(left_d, (x, 0))
				x += tile_w
				while x <= right_x - tile_w:
					platform_surf.blit(mid_d, (x, 0))
					x += tile_w
				platform_surf.blit(right_d, (right_x, 0))

			surface.blit(platform_surf, (draw_x, draw_y))
			if self.crack_lines_max:
				cracks = self._make_cracks(self.crack_lines_max, crack_alpha)
				mask = pygame.mask.from_surface(mask_surf)
				mask_surface = mask.to_surface(
					setcolor=(255, 255, 255, 255),
					unsetcolor=(0, 0, 0, 0),
				)
				cracks.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
				surface.blit(cracks, (draw_x, draw_y))
			return

		# Fallback to solid rects if tiles aren't loaded.
		platform_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
		color_with_alpha = (*self.color, alpha)
		pygame.draw.rect(
			platform_surf,
			color_with_alpha,
			(0, 0, self.w, self.h),
			border_radius=4,
		)
		draw_x = int(self.x)
		draw_y = int(self.y)
		surface.blit(platform_surf, (draw_x, draw_y))
		if self.crack_lines_max:
			cracks = self._make_cracks(self.crack_lines_max, crack_alpha)
			surface.blit(cracks, (draw_x, draw_y))

	def _make_cracks(self, count, alpha):
		"""Create a cracks overlay surface based on fade progress."""
		if count <= 0:
			return pygame.Surface((self.w, self.h), pygame.SRCALPHA)
		rng = random.Random(self.crack_seed)
		overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
		color = (40, 35, 35, alpha)
		segments = max(3, self.w // 8)
		min_x = 2
		max_x = max(2, self.w - 3)
		min_y = 2
		max_y = max(2, self.h - 3)
		for i in range(count):
			y = rng.randint(min_y, max_y)
			points = [(min_x, y)]
			for s in range(segments):
				x = min_x + int((s + 1) * (max_x - min_x) / segments)
				y += rng.randint(-3, 3)
				if y < min_y:
					y = min_y
				elif y > max_y:
					y = max_y
				points.append((x, y))
			pygame.draw.lines(overlay, color, False, points, 1)
		return overlay
