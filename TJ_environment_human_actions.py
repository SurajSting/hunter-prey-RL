"""
  * T&J's actions are determined by:
               N  W  S  E
  Ta[action] - 0, 1, 2, 3 
  Ja[action] - 4, 5, 6, 7
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
import sys
import numpy as np

from pycolab import ascii_art
from pycolab import human_ui
from pycolab import things as plab_things
from pycolab import rendering
from pycolab.prefab_parts import sprites as prefab_sprites

GAME_ART = [['#######', # 0  1  2  3  4  5  6
             '#     #', # 7  8  9  10 11 12 13
             '#   J #', # 14 15 16 17 18 19 20
             '# T   #', # 21 22 23
             '#     #',
             '#######',],

           ['#################',
            '#            xxx#',
            '#            xTx#',
            '#            xxx#',
            '#               #',
            '#               #',
            '#               #',
            '#               #',
            '#yyy            #',
            '#yJy            #',
            '#yyy            #',
            '#################']
            ]

COLOURS = {'#': (300, 300, 300),
           'T': (999, 0, 0),
           ' ': (500, 500, 500),
           'J': (0, 0, 999),
           'x': (400, 700, 200),
           'y': (400, 700, 200)}

def make_game(level):
  """Builds and returns a T&J game."""
  game_art = GAME_ART[level]


  update_schedule = [['J'],['T'],['y'],['x'],['#']]

  return ascii_art.ascii_art_to_game(
      game_art, what_lies_beneath=' ',
      sprites={'T': TomSprite, 'J':JerrySprite},
      drapes={'#': Walls, 'x': TomField, 'y': JerryField},
      update_schedule=update_schedule)

class Walls(plab_things.Drape):
  def __init__(self, curtain, character):
    super(Walls, self).__init__(curtain, character)

  def update(self, actions, board, layers, backdrop, things, the_plot):
    pass

class TomSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for our player. This `Sprite` ties actions to going in the  four cardinal directions.
  """
  def __init__(self, corner, position, character):
    """Inform superclass that we can't walk through walls."""
    super(TomSprite, self).__init__(
    corner, position, character, impassable='#')

  def update(self, actions, board, layers, backdrop, things, the_plot):	
    # Apply motion commands.
    if actions == 0:    # walk upward?
      self._north(board, the_plot)
    elif actions == 1:  # walk downward?
      self._west(board, the_plot)
    elif actions == 2:  # walk leftward?
      self._south(board, the_plot)
    elif actions == 3:  # walk rightward?
      self._east(board, the_plot)
    elif actions == 8:  # quit?
      the_plot.terminate_episode()

    Tr, Tc = things['T'].position

    if layers['y'][Tr][Tc] == True:
      reward = np.array([1, -1])
      the_plot.add_reward(reward)
      the_plot.terminate_episode()

class TomField(plab_things.Drape):
  def __init__(self, curtain, character):
    super(TomField, self).__init__(curtain, character)

  def update(self, actions, board, layers, backdrop, things, the_plot):
    tomPosition = things['T'].position
    Tr = tomPosition[0]
    Tc = tomPosition[1]
    xRow, xCol = np.where(layers['x'])
    # Clearing current x positions
    for pos in zip(xRow, xCol):
      listPos = list(pos)
      self.curtain[listPos[0]][listPos[1]] = False

    # Rebuilding x postions based on 'T'
    self.curtain[Tr-1][Tc-1] = True
    self.curtain[Tr-1][Tc] = True
    self.curtain[Tr-1][Tc+1] = True
    self.curtain[Tr][Tc-1] = True
    self.curtain[Tr][Tc+1] = True
    self.curtain[Tr+1][Tc-1] = True
    self.curtain[Tr+1][Tc] = True
    self.curtain[Tr+1][Tc+1] = True

class JerrySprite(prefab_sprites.MazeWalker):
  """A `Sprite` for our player.

  This `Sprite` ties actions to going in the four cardinal directions. If we
  reach a magical location (in this example, (4, 3)), the agent receives a
  reward of 1 and the epsiode terminates.
  """

  def __init__(self, corner, position, character):
    """Inform superclass that we can't walk through walls."""
    super(JerrySprite, self).__init__(
        corner, position, character, impassable='#')

  def update(self, actions, board, layers, backdrop, things, the_plot):
    # Apply motion commands.
    if actions == 4:    # walk upward?
      self._north(board, the_plot)
    elif actions == 5:  # walk downward?
      self._west(board, the_plot)
    elif actions == 6:  # walk leftward?
      self._south(board, the_plot)
    elif actions == 7:  # walk rightward?
      self._east(board, the_plot)
    elif actions == 8:  # quit?
      the_plot.terminate_episode()

    Jr, Jc = things['J'].position

    if layers['x'][Jr][Jc] == True:
      reward = np.array([1, -1])
      the_plot.add_reward(reward)
      the_plot.terminate_episode()

class JerryField(plab_things.Drape):
  def __init__(self, curtain, character):
    super(JerryField, self).__init__(curtain, character)

  def update(self, actions, board, layers, backdrop, things, the_plot):
    jerryPosition = things['J'].position
    Jr = jerryPosition[0]
    Jc = jerryPosition[1]
    yRow, yCol = np.where(layers['y'])
    # Clearing current x positions
    for pos in zip(yRow, yCol):
      listPos = list(pos)
      self.curtain[listPos[0]][listPos[1]] = False

    # Rebuilding y postions based on 'J'
    self.curtain[Jr-1][Jc-1] = True
    self.curtain[Jr-1][Jc] = True
    self.curtain[Jr-1][Jc+1] = True
    self.curtain[Jr][Jc-1] = True
    self.curtain[Jr][Jc+1] = True
    self.curtain[Jr+1][Jc-1] = True
    self.curtain[Jr+1][Jc] = True
    self.curtain[Jr+1][Jc+1] = True


def main(argv=()):
  game = make_game(int(argv[1]) if len(argv) > 1 else 0)

  # Make a CursesUi to play it with.
  ui = human_ui.CursesUi(
      keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 2,
                       curses.KEY_LEFT: 1, curses.KEY_RIGHT: 3,
                       'i': 4, 
                       'k': 6,
                       'j': 5,
                       'l': 7,
                       'q': 8},
      delay=0, colour_fg=COLOURS)

  # Let the game begin!
  ui.play(game)

if __name__ == '__main__':
  main(sys.argv)
