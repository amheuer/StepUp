"""Platform generation helpers."""

import random

from config import (
	MOVING_PLATFORM_CHANCE,
	PLATFORM_BASE_WIDTH,
	PLATFORM_HEIGHT,
	WINDOW_HEIGHT,
	WINDOW_WIDTH,
)
from entities import Platform


def create_initial_platforms():
	"""Generate the initial set of platforms."""
	platforms = []

	# Base platform at the bottom
	platforms.append(
		Platform(
			WINDOW_WIDTH // 2 - 40,
			WINDOW_HEIGHT - 40,
			width=80,
			height=14,
		)
	)

	# Generate platforms going upward
	y = WINDOW_HEIGHT - 120
	while y > -2000:
		x = random.randint(0, WINDOW_WIDTH - 60)
		is_moving = random.random() < MOVING_PLATFORM_CHANCE
		platforms.append(
			Platform(
				x,
				y,
				width=random.randint(50, 80),
				height=PLATFORM_HEIGHT,
				moving=is_moving,
			)
		)
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
		platforms.append(
			Platform(
				new_x,
				new_y,
				width=random.randint(50, 90),
				height=PLATFORM_HEIGHT,
				moving=is_moving,
			)
		)
