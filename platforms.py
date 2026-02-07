"""Platform generation helpers."""

import random

from config import (
	FRAGILE_PLATFORM_CHANCE,
	MAX_PLATFORM_GAP,
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
			WINDOW_WIDTH // 2 - PLATFORM_BASE_WIDTH // 2,
			WINDOW_HEIGHT - 40,
			width=PLATFORM_BASE_WIDTH,
			height=PLATFORM_HEIGHT,
		)
	)

	# Generate platforms going upward
	y = WINDOW_HEIGHT - 120
	while y > -2000:
		x = random.randint(0, WINDOW_WIDTH - 60)
		rand = random.random()
		if rand < MOVING_PLATFORM_CHANCE:
			kind = "moving"
		elif rand < MOVING_PLATFORM_CHANCE + FRAGILE_PLATFORM_CHANCE:
			kind = "fragile"
		else:
			kind = "normal"
		platforms.append(
			Platform(
				x,
				y,
				width=PLATFORM_BASE_WIDTH,
				height=PLATFORM_HEIGHT,
				kind=kind,
			)
		)
		gap_max = MAX_PLATFORM_GAP if kind != "moving" else 140
		y -= random.randint(60, gap_max)
		if len(platforms) > 40:
			break

	return platforms


def generate_new_platforms(platforms):
	"""Generate new platforms above existing ones. Returns newly created platforms."""
	new_platforms = []
	while len(platforms) < 10:
		top_y = min((p.y for p in platforms), default=0)
		new_x = random.randint(0, WINDOW_WIDTH - 60)
		rand = random.random()
		if rand < MOVING_PLATFORM_CHANCE:
			kind = "moving"
		elif rand < MOVING_PLATFORM_CHANCE + FRAGILE_PLATFORM_CHANCE:
			kind = "fragile"
		else:
			kind = "normal"
		gap_max = MAX_PLATFORM_GAP if kind != "moving" else 140
		new_y = top_y - random.randint(60, gap_max)
		new_platform = Platform(
			new_x,
			new_y,
			width=PLATFORM_BASE_WIDTH,
			height=PLATFORM_HEIGHT,
			kind=kind,
		)
		platforms.append(new_platform)
		new_platforms.append(new_platform)
	return new_platforms
