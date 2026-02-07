"""Collision detection helpers."""


def check_if_on_platform(player, platforms, prev_y):
	"""
	Check if player is currently resting on a platform.
	Returns the platform the player is on, or None.
	Only detects platforms when player is falling or stationary (not jumping upward).
	Also detects if player is stuck inside platform (emergency case).
	"""
	closest_platform = None
	closest_distance = float("inf")
	prev_bottom = prev_y + player.radius

	for platform in platforms:
		if not platform.is_active:
			continue

		# One-way platforms: ignore collisions while moving upward.
		if player.vy < 0:
			continue

		player_left = player.x - player.radius
		player_right = player.x + player.radius
		platform_left = platform.x
		platform_right = platform.x + platform.w

		# Check if player's bounding box overlaps with platform horizontally
		if player_right > platform_left and player_left < platform_right:
			# Check if player is above or on the platform (player's bottom near platform's top)
			player_bottom = player.y + player.radius
			platform_top = platform.rect.y
			platform_bottom = platform.rect.y + platform.rect.h

			# Normal case: player crosses the platform top while falling
			if (
				player.vy >= 0
				and prev_bottom <= platform_top + 2
				and player_bottom >= platform_top - 2
			):
				distance = player_bottom - platform_top
				if distance < closest_distance:
					closest_distance = distance
					closest_platform = platform
			# Emergency case: if player is stuck inside platform, snap out based on prior position
			elif player_bottom > platform_top and player.y < platform_bottom:
				if prev_y <= platform_top:
					player.y = platform_top - player.radius
					return platform
				if prev_y >= platform_bottom:
					player.y = platform_bottom + player.radius
					return None

	return closest_platform
