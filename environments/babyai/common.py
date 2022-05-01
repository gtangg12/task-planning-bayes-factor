def index_dict(x):
    return dict(zip(x, range(len(x))))


""" Minigrid specs """
#
VIEW_SHAPE = (7, 7)

# 0 denotes pointing right (positive X) globally and directions cycle clockwise
NUM_DIRECTIONS = 4
DIRECTIONS = ['right', 'down', 'left', 'up']
DIRECTIONS_TO_INDEX = index_dict(DIRECTIONS)

# 
ACTIONS = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']
ACTIONS_TO_INDEX = index_dict(ACTIONS)

NUM_ACTIONS = len(ACTIONS)

#
OBJECTS = ['unseen', 'empty', 'wall', 'floor', 'door',
           'key', 'ball', 'box', 'goal', 'lava', 'agent']
COLORS = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
# open door also used when no door present in tile
DOOR_STATES = ['open', 'closed', 'locked']

NUM_VIEW_FEATURES_LIST = [len(OBJECTS), len(COLORS), len(DOOR_STATES)]
NUM_VIEW_FEATURES = sum(NUM_VIEW_FEATURES_LIST)

# baby language has 31 words
VOCAB = ['then', 'after', 'you', 'and', 'go', 'to', 'pick', 'up', 'open', 'put',
         'next', 'door', 'ball', 'box', 'key', 'on', 'your', 'left', 'right',
         'in', 'front', 'of', 'behind', 'red', 'green', 'blue', 'purple',
         'yellow', 'grey', 'the', 'a']
VOCAB_TO_INDEX = index_dict(VOCAB)

#
TASK_TYPES = ['BossLevel', 'GoTo', 'GoToImpUnlock', 'GoToLocal', 'GoToObj', 'GoToObjMaze', 'GoToObjMazeOpen',
              'GoToRedBall', 'GoToRedBallGrey', 'GoToSeq', 'Open', 'Pickup', 'PickupLoc', 'PutNext',
              'PutNextLocal', 'Synth', 'SynthLoc', 'SynthSeq', 'UnblockPickup', 'Unlock']
TASK_TYPES_TO_INDEX = index_dict(TASK_TYPES)

#
TASK_GROUPS = {
    'easy': {'GoToLocal', 'GoToObj', 'GoToRedBall', 'GoToRedBallGrey', 'PickupLoc', 'PutNextLocal'},
    'all': set(TASK_TYPES),
}


""" Data Specs """
FEATURE_NAMES = ['task', 'images', 'directions', 'actions']
