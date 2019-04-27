import pygame
import Game
import Food
import math
import random
from Action import *
from Constants import *

g = Game.Game(numberOfSnakes, gridSize, globalEpisodeLength) #Instantiating an object of class Game
scalingFactor = 9 #Scaling the size of the grid

WIDTH = scalingFactor*gridSize
HEIGHT = scalingFactor*gridSize
pos_score_x = math.floor(WIDTH/(numberOfSnakes +1)) # For displaying score
pos_score_y = math.floor(HEIGHT/20)
font_size = math.floor(HEIGHT/40)
black = (0, 0, 0)
white = (255, 255, 255)
red = (200, 0, 0)
green = (0, 200, 0)
pygame.init() # Initializes all pygame modules
font_name = pygame.font.match_font('arial')

def draw_text(surf, text, size,x,y):
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, black)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surf.blit(text_surface,text_rect)

def to_pygame(p):
    """Convert coordinates into pygame coordinates (lower-left => top left)."""
    return (p.x, gridSize-p.y)


def manual_action(g):
    """
    This initialization may be a bit computation intensive, but added this line here for representation of two snakes
    Else can simply initialize to 0
    """
    if numberOfSnakes>1:
        actionsList = [random.choice(g.snakes[x].permissible_actions()) for x in range(numberOfSnakes)]
    else:
        actionsList = [0]

    keys = pygame.key.get_pressed()
    if g.snakes[0].joints == []:
        defaultaction = g.snakes[0].findDirection(g.snakes[0].head, g.snakes[0].end)
    else:
        defaultaction = g.snakes[0].findDirection(g.snakes[0].head, g.snakes[0].joints[0])
    
    if event.type == pygame.KEYDOWN:
        if keys[pygame.K_RIGHT]:
            actionsList[0] = Action.RIGHT
        elif keys[pygame.K_LEFT]:
            actionsList[0] = Action.LEFT
        elif keys[pygame.K_UP]:
            actionsList[0] = Action.TOP
        elif keys[pygame.K_DOWN]:
            actionsList[0] = Action.DOWN
    else:
        actionsList[0] = defaultaction
    return actionsList         

crashed = False
win = pygame.display.set_mode((scalingFactor * gridSize, scalingFactor * gridSize))  # Game Window
screen = pygame.Surface((gridSize, gridSize))  # Grid Screen
pygame.display.set_caption("Snake Game")
clock = pygame.time.Clock()

while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True
    screen.fill(white)
    for idx in range(numberOfSnakes):
        s = g.snakes[idx]
        body = [s.head]
        body.extend(s.joints)
        body.append(s.end)
        for p in Food.foodList:
            pygame.draw.line(screen, green, to_pygame(p), to_pygame(p), 1)  # Drawing all the food points
        for i in range(len(body) - 1):
            pygame.draw.line(screen, black, to_pygame(body[i]), to_pygame(body[i + 1]), 1) # Drawing Lines for the snake's body
        pygame.draw.line(screen, red, to_pygame(body[0]),to_pygame(body[0]),1) # To draw the red pixel of the head
        actionsList = manual_action(g)
        """
        actionsList = rl_agent(g)
        Can also add an if condition here to have one player and one agent
        and then append the two lists obtained, to pass it to move method
        """
        g.move(actionsList)
    win.blit(pygame.transform.scale(screen, win.get_rect().size), (0, 0)) # Transforms the screen window into the win window
    for idx in range(numberOfSnakes):
        draw_text(win, "Snake" + str(idx) + "  " + str(g.snakes[idx].score), font_size, pos_score_x * (idx + 1), pos_score_y) #Displaying score
    pygame.display.update()
    clock.tick(10)  #means that for every second at most 10 frames should pass.
pygame.quit()