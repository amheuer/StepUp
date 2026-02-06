import pygame
import random
import sys


# Simple Doodle Jump clone using Pygame
# - Player is a circle
# - Platforms are rectangles
# - No external assets required


WIDTH, HEIGHT = 400, 600
FPS = 60


class Player:
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.radius = 16
		self.vx = 0
		self.vy = 0
		self.color = (30, 120, 200)

	@property
	def rect(self):
		return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)

	def update(self, dt, keys):
		# Horizontal movement
		speed = 200
		ax = 0
		if keys[pygame.K_LEFT] or keys[pygame.K_a]:
			ax -= 1
		if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
			ax += 1
		self.vx = ax * speed

		# Apply velocities
		self.x += self.vx * dt
		self.y += self.vy * dt

		# Gravity
		self.vy += 800 * dt

		# Wrap around horizontally
		if self.x < -self.radius:
			self.x = WIDTH + self.radius
		elif self.x > WIDTH + self.radius:
			self.x = -self.radius

	def jump(self, strength=420):
		self.vy = -strength

	def draw(self, surf):
		pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), self.radius)


class Platform:
	def __init__(self, x, y, w=60, h=12, moving=False):
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.color = (50, 200, 110)
		self.moving = moving
		self.dir = 1 if random.random() < 0.5 else -1
		self.speed = random.uniform(30, 70) if moving else 0

	@property
	def rect(self):
		return pygame.Rect(self.x, self.y, self.w, self.h)

	def update(self, dt):
		if self.moving:
			self.x += self.dir * self.speed * dt
			if self.x < 0:
				self.x = 0
				self.dir *= -1
			elif self.x + self.w > WIDTH:
				self.x = WIDTH - self.w
				self.dir *= -1

	def draw(self, surf):
		pygame.draw.rect(surf, self.color, self.rect, border_radius=4)


def create_initial_platforms():
	platforms = []
	# base platform under the player
	platforms.append(Platform(WIDTH // 2 - 40, HEIGHT - 40, w=80, h=14))
	# random platforms
	y = HEIGHT - 120
	while y > -2000:
		x = random.randint(0, WIDTH - 60)
		platforms.append(Platform(x, y, w=random.randint(50, 80), h=12, moving=(random.random() < 0.12)))
		y -= random.randint(60, 120)
		if len(platforms) > 40:
			break
	return platforms


def main():
	pygame.init()
	screen = pygame.display.set_mode((WIDTH, HEIGHT))
	pygame.display.set_caption("StepUp")
	clock = pygame.time.Clock()
	font = pygame.font.SysFont(None, 32)

	player = Player(WIDTH // 2, HEIGHT - 80)
	platforms = create_initial_platforms()

	score = 0
	high_score = 0

	running = True
	game_over = False

	while running:
		dt = clock.tick(FPS) / 1000.0
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					running = False
				if game_over and event.key == pygame.K_r:
					# restart
					player = Player(WIDTH // 2, HEIGHT - 80)
					platforms = create_initial_platforms()
					score = 0
					game_over = False

		if not game_over:
			keys = pygame.key.get_pressed()
			
			# Store previous position for collision detection
			prev_y = player.y
			prev_x = player.x
			
			player.update(dt, keys)

			# Update platforms
			for p in platforms:
				p.update(dt)

			# Collision: only when falling and moving downward
			if player.vy > 0:
				closest_platform = None
				closest_distance = float('inf')
				
				for p in platforms:
					# Create a rect for the player's previous position
					prev_rect = pygame.Rect(prev_x - player.radius, prev_y - player.radius, 
											player.radius * 2, player.radius * 2)
					
					# Check if player was NOT colliding before but IS colliding now
					if not prev_rect.colliderect(p.rect) and player.rect.colliderect(p.rect):
						# This is a fresh collision - find the closest platform
						distance = player.y + player.radius - p.y
						if distance < closest_distance:
							closest_distance = distance
							closest_platform = p
				
				if closest_platform:
					player.y = closest_platform.y - player.radius
					player.jump()

			# Scroll: if player goes above threshold, move everything down
			scroll_threshold = HEIGHT * 0.4
			if player.y < scroll_threshold:
				dy = scroll_threshold - player.y
				player.y = scroll_threshold
				# move platforms down
				for p in platforms:
					p.y += dy
				score += int(dy)

			# Remove platforms that are below the screen
			platforms = [p for p in platforms if p.y < HEIGHT + 50]

			# Generate new platforms above
			while len(platforms) < 10:
				top_y = min((p.y for p in platforms), default=0)
				new_y = top_y - random.randint(60, 140)
				new_x = random.randint(0, WIDTH - 60)
				platforms.append(Platform(new_x, new_y, w=random.randint(50, 90), moving=(random.random() < 0.12)))

			# Game over if player falls below bottom
			if player.y - player.radius > HEIGHT:
				game_over = True
				if score > high_score:
					high_score = score

		# Drawing
		screen.fill((15, 18, 30))

		# Parallax-ish stars: simple background dots based on score
		for i in range(30):
			sx = (i * 37 + score) % WIDTH
			sy = (i * 53 + score // 3) % HEIGHT
			screen.set_at((sx, sy), (200, 200, 220))

		for p in platforms:
			p.draw(screen)

		player.draw(screen)

		score_surf = font.render(f"Score: {score}", True, (230, 230, 230))
		screen.blit(score_surf, (8, 8))

		hs_surf = font.render(f"High: {high_score}", True, (200, 200, 200))
		screen.blit(hs_surf, (WIDTH - hs_surf.get_width() - 8, 8))

		if game_over:
			over_surf = font.render("Game Over - Press R to restart", True, (255, 200, 200))
			screen.blit(over_surf, ((WIDTH - over_surf.get_width()) // 2, HEIGHT // 2 - 20))

		pygame.display.flip()

	pygame.quit()
	sys.exit()


if __name__ == '__main__':
	main()
