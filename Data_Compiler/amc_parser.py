"""
Copyright @c Sadeghi
mahdi.sadeghi@uni-tuebingen.de
"""
import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import euler2mat
from mpl_toolkits.mplot3d import Axes3D


class Joint:
  def __init__(self, name, direction, length, axis, dof, limits):
    """
    Definition of basic joint. The joint also contains the information of the
    bone between it's parent joint and itself.
    for detailed description for asf files.

    Parameter
    ---------
    name: Name of the joint defined in the asf file. There should always be one
    root joint. String.

    direction: Default direction of the joint(bone). The motions are all defined
    based on this default pose.

    length: Length of the bone.

    axis: Axis of rotation for the bone.

    dof: Degree of freedom. Specifies the number of motion channels and in what
    order they appear in the AMC file.

    limits: Limits on each of the channels in the dof specification

    """
    self.name = name
    self.direction = np.reshape(direction, [3, 1])
    self.length = length
    axis = np.deg2rad(axis)
    self.C = euler2mat(*axis)
    self.Cinv = np.linalg.inv(self.C)
    self.limits = np.zeros([3, 2])
    for lm, nm in zip(limits, dof):
      if nm == 'rx':
        self.limits[0] = lm
      elif nm == 'ry':
        self.limits[1] = lm
      else:
        self.limits[2] = lm
    self.parent = None
    self.children = []
    self.coordinate = None
    self.matrix = None

  def set_motion(self, motion):
    if self.name == 'root':
      self.coordinate = np.reshape(np.array([0, 0, 0]), [3, 1])
      rotation = np.deg2rad(motion['root'][3:])
      self.matrix = self.C.dot(euler2mat(*rotation)).dot(self.Cinv)
    else:
      idx = 0
      rotation = np.zeros(3)
      for axis, lm in enumerate(self.limits):
        if not np.array_equal(lm, np.zeros(2)):
          rotation[axis] = motion[self.name][idx]
          idx += 1
      rotation = np.deg2rad(rotation)
      self.matrix = self.parent.matrix.dot(self.C).dot(euler2mat(*rotation)).dot(self.Cinv)
      self.coordinate = self.parent.coordinate + self.length * self.matrix.dot(self.direction)
    for child in self.children:
      child.set_motion(motion)

  def draw(self):
    joints = self.to_dict()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(-50, 10)
    ax.set_ylim3d(-20, 40)
    ax.set_zlim3d(-20, 40)

    xs, ys, zs = [], [], []
    for joint in joints.values():
      xs.append(joint.coordinate[0, 0])
      ys.append(joint.coordinate[1, 0])
      zs.append(joint.coordinate[2, 0])
    plt.plot(zs, xs, ys, 'b.')

    for joint in joints.values():
      child = joint
      if child.parent is not None:
        parent = child.parent
        xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
        ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
        zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
        plt.plot(zs, xs, ys, 'r')

  def to_dict(self):
    ret = {self.name: self}
    for child in self.children:
      ret.update(child.to_dict())
    return ret

  def pretty_print(self):
    print('===================================')
    print('joint: %s' % self.name)
    print('direction:')
    print(self.direction)
    print('limits:', self.limits)
    print('parent:', self.parent)
    print('children:', self.children)

def get_cmap(n, name='hsv'):
  '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
  RGB color; the keyword argument name must be a standard mpl colormap name.'''
  return plt.cm.get_cmap(name, n)


def read_line(stream, idx):
  if idx >= len(stream):
    return None, idx
  line = stream[idx].strip().split()
  idx += 1
  return line, idx


def parse_asf(file_path):
  '''read joint data only'''
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    # meta infomation is ignored
    if line == ':bonedata':
      content = content[idx+1:]
      break

  # read joints
  joints = {'root': Joint('root', np.zeros(3), 0, np.zeros(3), [], [])}
  idx = 0
  while True:
    # the order of each section is hard-coded
    line, idx = read_line(content, idx)

    if line[0] == ':hierarchy':
      break

    assert line[0] == 'begin'

    line, idx = read_line(content, idx)
    assert line[0] == 'id'

    line, idx = read_line(content, idx)
    assert line[0] == 'name'
    name = line[1]

    line, idx = read_line(content, idx)
    assert line[0] == 'direction'
    direction = np.array([float(axis) for axis in line[1:]])

    # skip length
    line, idx = read_line(content, idx)
    assert line[0] == 'length'
    length = float(line[1])

    line, idx = read_line(content, idx)
    assert line[0] == 'axis'
    assert line[4] == 'XYZ'

    axis = np.array([float(axis) for axis in line[1:-1]])

    dof = []
    limits = []

    line, idx = read_line(content, idx)
    if line[0] == 'dof':
      dof = line[1:]
      for i in range(len(dof)):
        line, idx = read_line(content, idx)
        if i == 0:
          assert line[0] == 'limits'
          line = line[1:]
        assert len(line) == 2
        mini = float(line[0][1:])
        maxi = float(line[1][:-1])
        limits.append((mini, maxi))

      line, idx = read_line(content, idx)

    assert line[0] == 'end'
    joints[name] = Joint(
      name,
      direction,
      length,
      axis,
      dof,
      limits
    )

  # read hierarchy
  assert line[0] == ':hierarchy'

  line, idx = read_line(content, idx)

  assert line[0] == 'begin'

  while True:
    line, idx = read_line(content, idx)
    if line[0] == 'end':
      break
    assert len(line) >= 2
    for joint_name in line[1:]:
      joints[line[0]].children.append(joints[joint_name])
    for nm in line[1:]:
      joints[nm].parent = joints[line[0]]

  return joints


def parse_amc(file_path):
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    if line == ':DEGREES':
      content = content[idx+1:]
      break

  frames = []
  idx = 0
  line, idx = read_line(content, idx)
  assert line[0].isnumeric(), line
  EOF = False
  while not EOF:
    joint_degree = {}
    while True:
      line, idx = read_line(content, idx)
      if line is None:
        EOF = True
        break
      if line[0].isnumeric():
        break
      joint_degree[line[0]] = [float(deg) for deg in line[1:]]
    frames.append(joint_degree)
  return frames


def test_all(user_asf_path, user_amc_path, user_input_frames, user_input_features):
  asf_path = user_asf_path
  print('parsing %s' % asf_path)
  joints = parse_asf(asf_path)
  motions = parse_amc(user_amc_path)
  joints['root'].set_motion(motions[200])
  joints['root'].draw()
  input_frames = user_input_frames
  visual_input = np.zeros((user_input_features, input_frames, 3))
  for i in range(input_frames):
    perecived_joint = 0
    joints['root'].set_motion(motions[i])
    for joint in joints.values():
      if perecived_joint == 0:
        perecived_joint = perecived_joint + 1
        continue
      visual_input[perecived_joint-1, i, :] = joint.coordinate[:, 0]
      perecived_joint = perecived_joint +1

  print('Parsing is Done!')
  print(visual_input.shape)

  maxiX = []
  miniX = []
  maxiY = []
  miniY = []
  maxiZ = []
  miniZ = []
  for j7 in range(30):
      maxisoluX = np.max(visual_input[j7, :, 0], axis=0)
      maxisoluY = np.max(visual_input[j7, :, 1], axis=0)
      maxisoluZ = np.max(visual_input[j7, :, 2], axis=0)
      minisoluX = np.min(visual_input[j7, :, 0], axis=0)
      minisoluY = np.min(visual_input[j7, :, 1], axis=0)
      minisoluZ = np.min(visual_input[j7, :, 2], axis=0)
      maxiX.append(maxisoluX)
      miniX.append(minisoluX)
      maxiY.append(maxisoluY)
      miniY.append(minisoluY)
      maxiZ.append(maxisoluZ)
      miniZ.append(minisoluZ)
  maxiX = np.asarray(maxiX)
  maxiY = np.asarray(maxiY)
  maxiZ = np.asarray(maxiZ)
  miniX = np.asarray(miniX)
  miniY = np.asarray(miniY)
  miniZ = np.asarray(miniZ)

  max_allX = np.max(maxiX)
  max_allY = np.max(maxiY)
  max_allZ = np.max(maxiZ)
  min_allX = np.min(miniX)
  min_allY = np.min(miniY)
  min_allZ = np.min(miniZ)

  print('X is between  {nix} and {mix}'.format(mix=max_allX, nix=min_allX))
  print('Y is between {niY} and {miY}'.format(miY=max_allY, niY=min_allY))
  print('Z is between {niZ} and {miZ}'.format(miZ=max_allZ, niZ=min_allZ))

  # Here I would need to reduce the number of joints to 15
  nr_final_selected_joitns = 15
  selector = np.zeros(nr_final_selected_joitns)
  #selected joints are: Lhipjoint, Lfemur, Lfoot, Rhipjoint, Rfemur, Rfoot, Throax, Lowerneck, Head, Lclavicle, Lradius, Lhand, Rclavicle, Rradius, Rhand
  selector = [0, 1, 3, 5, 6, 8, 12, 13, 15, 16, 18, 20, 23, 25, 27]
  Joints_list_all = []
  for key in joints.keys():
    Joints_list_all.append(key)

  selected_joint_names = []
  for sj in selector:
    selected_joint_names.append(Joints_list_all[sj+1])

  visual_input = visual_input[selector, :, :]

  return visual_input, selected_joint_names
