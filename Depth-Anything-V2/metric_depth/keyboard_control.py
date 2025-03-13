import pygame
import requests
import json

# URL of the drone control server (adjust as needed)
SERVER_URL = 'http://127.0.0.1:5000/command'

# Pygame initialization
pygame.init()
screen = pygame.display.set_mode((600, 200))
pygame.display.set_caption("Drone Control Client")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

# Speed and default command values
DEFAULT_DISTANCE = 30  # centimeters
DEFAULT_ANGLE = 90     # degrees

running = True

def send_command(command_data):
    try:
        response = requests.post(SERVER_URL, json=command_data)
        print("Response:", response.text)
    except Exception as e:
        print("Error sending command:", e)

# Display instructions on the Pygame window
def draw_instructions():
    screen.fill((0, 0, 0))
    instructions = [
        "T: Takeoff",
        "L: Land",
        "Arrow UP/DOWN: Move Forward/Backward",
        "Arrow LEFT/RIGHT: Move Left/Right",
        "W/S: Move Up/Down",
        "A/D: Rotate Counter/Clockwise",
        "F: Flip Forward",
        "B: Flip Backward",
        "Q: Flip Left",
        "E: Flip Right",
    ]
    for i, line in enumerate(instructions):
        text_surface = font.render(line, True, (255, 255, 255))
        screen.blit(text_surface, (20, 20 + i * 20))
    pygame.display.flip()

while running:
    draw_instructions()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            # Drone takeoff and land
            if event.key == pygame.K_t:
                send_command({"action": "takeoff"})
            elif event.key == pygame.K_l:
                send_command({"action": "land"})
            # Movement commands
            elif event.key == pygame.K_UP:
                send_command({"action": "move", "direction": "forward", "distance": DEFAULT_DISTANCE})
            elif event.key == pygame.K_DOWN:
                send_command({"action": "move", "direction": "backward", "distance": DEFAULT_DISTANCE})
            elif event.key == pygame.K_LEFT:
                send_command({"action": "move", "direction": "left", "distance": DEFAULT_DISTANCE})
            elif event.key == pygame.K_RIGHT:
                send_command({"action": "move", "direction": "right", "distance": DEFAULT_DISTANCE})
            elif event.key == pygame.K_w:
                send_command({"action": "move", "direction": "up", "distance": DEFAULT_DISTANCE})
            elif event.key == pygame.K_s:
                send_command({"action": "move", "direction": "down", "distance": DEFAULT_DISTANCE})
            # Rotation commands
            elif event.key == pygame.K_a:
                send_command({"action": "rotate", "direction": "counter_clockwise", "angle": DEFAULT_ANGLE})
            elif event.key == pygame.K_d:
                send_command({"action": "rotate", "direction": "clockwise", "angle": DEFAULT_ANGLE})
            # Flip commands
            elif event.key == pygame.K_f:
                send_command({"action": "flip", "direction": "forward"})
            elif event.key == pygame.K_b:
                send_command({"action": "flip", "direction": "backward"})
            elif event.key == pygame.K_q:
                send_command({"action": "flip", "direction": "left"})
            elif event.key == pygame.K_e:
                send_command({"action": "flip", "direction": "right"})

    # clock.tick(30)

pygame.quit()
