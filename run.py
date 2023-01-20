import pygame
import numpy as np
from RK import RK4


SIM_DIMENSION = 600
BUFFER = 150
SIZE = SIM_DIMENSION + BUFFER
TIME_STEP = 0.02
WHITE = pygame.Color("white")
BLACK = pygame.Color("black")
RED = pygame.Color("red")
BLUE = pygame.Color("blue")

pygame.init()
pygame.display.set_caption("Double Pendulum Sim")
clock = pygame.time.Clock()


class Sim:
    def __init__(self, *args):
        self.screen = pygame.display.set_mode((SIZE, SIZE))
        self.trace = Trace()
        self.mass1 = Mass(RED, 15)
        self.mass2 = Mass(RED, 15)
        self.th1, self.th2, self.th1d, self.th2d, self.g, self.m1, self.m2, self.l1, self.l2 = args
        self.time_scale = 1
        self.paused = True
        self.adjust_mode = 0
        self.bg = WHITE

        self.setup()
        self.main()
    
    def setup(self):
        self.step = 0
        self.trace.screen.fill(BLACK)

    def main(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self.handle_keyboard(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_click(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_mouse_drag(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.handle_mouse_up()
            
            self.screen.fill(self.bg)
            self.draw()
            if not self.paused:
                self.compute_next()
                self.step += 1

            clock.tick(self.time_scale/TIME_STEP)
            pygame.display.flip()

    def compute_next(self):
        def F1(th1, th2, th1d, th2d): # th1d
            return th1d
        
        def F2(th1, th2, th1d, th2d): # th2d
            return th2d

        def G1(th1, th2, th1d, th2d): # th1dd
            return (-self.g*(2*self.m1 + self.m2)*np.sin(th1) - self.m2*self.g*np.sin(th1 - 2*th2) - 2*np.sin(th1 - th2)*self.m2*(th2d**2*self.l2 + 
                    th1d**2*self.l1*np.cos(th1 - th2))) / (self.l1*(2*self.m1 + self.m2 - self.m2*np.cos(2*(th1 - th2))))

        def G2(th1, th2, th1d, th2d): # th2dd
            return (2*np.sin(th1 - th2)*(th1d**2*self.l1*(self.m1 + self.m2) + self.g*(self.m1 + self.m2)*np.cos(th1) + 
                th2d**2*self.l2*self.m2*np.cos(th1 - th2))) / (self.l2*(2*self.m1 + self.m2 - self.m2*np.cos(2*(th1 - th2))))

        self.th1prev, self.th2prev, self.th1dprev, self.th2dprev = self.th1, self.th2, self.th1d, self.th2d
        self.th1, self.th2, self.th1d, self.th2d = RK4(F1, G1, F2, G2, self.th1, self.th1d, self.th2, self.th2d, TIME_STEP)
        
    def get_cartesian(self, th1, th2):
        scale = SIM_DIMENSION / (2*(self.l1 + self.l2))
        x1 = SIZE/2 + scale * self.l1*np.sin(th1)
        x2 = x1 + scale * self.l2*np.sin(th2)
        y1 = SIZE/2 + scale * self.l1*np.cos(th1)
        y2 = y1 + scale * self.l2*np.cos(th2)
        return x1, y1, x2, y2

    def draw(self):
        x1, y1, x2, y2 = self.get_cartesian(self.th1, self.th2)

        if self.trace.on and self.step > 0:
            x2prev, y2prev = self.get_cartesian(self.th1prev, self.th2prev)[2:]
            if not self.adjust_mode:
                self.trace.update(x2, y2, x2prev, y2prev)
            self.trace.draw(self.screen)

        pygame.draw.line(self.screen, BLACK, (SIZE/2, SIZE/2), (x1, y1), 5)
        pygame.draw.line(self.screen, BLACK, (x1, y1), (x2, y2), 5)
        self.mass1.rect.centerx, self.mass1.rect.centery = x1, y1
        self.mass2.rect.centerx, self.mass2.rect.centery = x2, y2
        self.mass1.draw(self.screen)
        self.mass2.draw(self.screen)
        

    def handle_keyboard(self, key):
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_t:
            self.trace.on = not self.trace.on
        elif key == pygame.K_r:
            self.trace.screen.fill(BLACK)

    def handle_mouse_click(self, mousepos):
        for i, mass in enumerate((self.mass1, self.mass2)):
            if mass.rect.collidepoint(mousepos):
                self.paused = True
                self.trace.screen.fill(BLACK)
                if i == 0: 
                    self.adjust_mode = 1
                else:
                    self.adjust_mode = 2
    
    def handle_mouse_drag(self, mousepos):
        mouse_x, mouse_y = mousepos
        mouse_x -= SIZE/2
        mouse_y -= SIZE/2
        if self.adjust_mode == 1:
            self.th1 = np.arctan2(-mouse_y, mouse_x) + np.pi/2
        elif self.adjust_mode == 2:
            x1, y1 = self.get_cartesian(self.th1, self.th2)[:2]
            x1 -= SIZE/2
            y1 -= SIZE/2
            self.th2 = np.arctan2(y1 - mouse_y, mouse_x - x1) + np.pi/2
    
    def handle_mouse_up(self):
        if self.adjust_mode:
            self.th1d, self.th2d, self.step, self.adjust_mode = 0, 0, 0, 0

class Trace:
    def __init__(self):
        self.screen = pygame.Surface((SIZE, SIZE))
        self.screen.set_colorkey(BLACK) # Makes black background transparent
        self.on = False

    def update(self, xcurr, ycurr, xprev, yprev):
        pygame.draw.line(self.screen, BLUE, (xcurr, ycurr), (xprev, yprev), 3)
    
    def draw(self, screen):
        screen.blit(self.screen, (0, 0))

class Mass(pygame.sprite.Sprite):
    def __init__(self, color, radius):
        super().__init__()

        self.image = pygame.Surface((2*radius, 2*radius))
        self.image.set_colorkey(BLACK)
        pygame.draw.circle(self.image, color, (radius, radius), radius)
        self.rect = self.image.get_rect()
    
    def draw(self, screen):
        screen.blit(self.image, self.rect)


if __name__ == "__main__":
    Sim(np.pi/2, np.pi/2, 0, 0, 9.81, 1, 1, 1, 1) # Default initial conds/parameters