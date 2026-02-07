"""Collectable coin entities and helpers."""

import random
import pygame

from config import (
	COIN_COLOR,
	COIN_RADIUS,
	COIN_SPAWN_CHANCE,
	COIN_VALUE,
	WINDOW_HEIGHT,
	WINDOW_WIDTH,
)


class Coin:
	"""Represents a collectable coin."""

	frames = []
	frame_time_ms = 80
	frames_loaded = False

	@classmethod
	def load_frames(cls, frames, frame_time_ms=80):
		cls.frames = frames
		cls.frame_time_ms = frame_time_ms
		cls.frames_loaded = len(frames) > 0

	def __init__(self, x, y, radius=COIN_RADIUS):
		self.x = x
		self.y = y
		self.radius = radius
		self.color = COIN_COLOR
		self.collected = False
		self.anim_offset = random.randint(0, 1000)

	def draw(self, surface):
		"""Draw the coin as a simple circle or animated sprite."""
		if self.collected:
			return
		if Coin.frames_loaded:
			t = pygame.time.get_ticks() + self.anim_offset
			index = (t // Coin.frame_time_ms) % len(Coin.frames)
			frame = Coin.frames[index]
			rect = frame.get_rect(center=(int(self.x), int(self.y)))
			surface.blit(frame, rect)
			return
		pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)


def spawn_coins_near_platforms(platforms):
	"""Spawn coins near platforms with a fixed chance."""
	coins = []
	for platform in platforms:
		if random.random() > COIN_SPAWN_CHANCE:
			continue

		coin_x = random.randint(int(platform.x + 6), int(platform.x + platform.w - 6))
		coin_y = platform.y - random.randint(25, 45)
		if coin_y < -50:
			continue
		coins.append(Coin(coin_x, coin_y))

	return coins


def collect_coins(player, coins):
	"""Collect coins that overlap the player. Returns (score_gained, count_gained)."""
	score_gained = 0
	count_gained = 0
	for coin in coins:
		if coin.collected:
			continue
		dx = player.x - coin.x
		dy = player.y - coin.y
		distance_sq = dx * dx + dy * dy
		if distance_sq <= (player.radius + coin.radius) ** 2:
			coin.collected = True
			score_gained += COIN_VALUE
			count_gained += 1

	return score_gained, count_gained


def cull_coins(coins):
	"""Remove collected or off-screen coins."""
	return [c for c in coins if not c.collected and c.y < WINDOW_HEIGHT + 50]
