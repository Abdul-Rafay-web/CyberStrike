from __future__ import annotations

import logging

import pygame

from config import CONFIG

log = logging.getLogger(__name__)


def run_window() -> None:
    """Open a blank window and keep it alive until a user closes it."""
    pygame.init()
    theme = CONFIG.theme
    screen = pygame.display.set_mode(theme.window_size)
    pygame.display.set_caption("CyberStrike")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False

        screen.fill(theme.background)
        pygame.display.flip()
        clock.tick(theme.fps)

    pygame.quit()
    log.info("window closed")
