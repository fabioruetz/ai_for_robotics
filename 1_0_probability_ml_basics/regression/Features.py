from abc import ABCMeta, abstractmethod
import math
import numpy as np


class Feature():
  """
  Feature base class.
  """
  __metaClass__ = ABCMeta

  @abstractmethod
  def evaluate(self, x1, x2, x3 = 0, x4 = 0):
    pass

# Feature classes
class LinearX1(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return x1

class LinearX2(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return x2

class LinearX3(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return x3

class LinearX4(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return x4

class SquareX1(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return x1**2

class SquareX2(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return x2**2

class SquareX3(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return x3 ** 2

class SquareX4(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return x4**2

# Cross Terms
class CrossTermX1X2(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return x1 * x2

class CrossTermX1X3(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return x1 * x3

class CrossTermX1X4(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return x1 * x4

class CrossTermX2X3(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return x2 * x3

class CrossTermX2X4(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return x2 * x4

class CrossTermX3X4(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return x3 * x4

# Exponential Features
class ExpX1(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return math.exp(x1)

class ExpX2(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return math.exp(x2)

class ExpX3(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return math.exp(x3)

class ExpX4(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return math.exp(x4)

# Log Features
class LogX1(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return math.log(x1)


class LogX2(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return math.log(x2)

class LogX3(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return math.log(x3)

class LogX4(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return math.log(x4)

# Trigonometric Features
class SinX1(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return math.sin(x1)

class SinX2(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return math.sin(x2)

class SinX3(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return math.sin(x3)

class SinX4(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return math.sin(x4)

class CosX1(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return math.cos(x1)

class CosX2(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return math.cos(x2)

class CosX3(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return math.cos(x3)

class CosX4(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return math.cos(x4)
# Identity Features

class Identity(Feature):
  def evaluate(self, x1, x2, x3, x4):
    return 1
