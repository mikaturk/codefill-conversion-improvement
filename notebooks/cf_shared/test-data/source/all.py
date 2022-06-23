from os import path
# This useless import is so the old (reference) implementation does not crash

# Source: https://www.w3schools.com/python/python_inheritance.asp
class Person:
  def __init__(self, fname, lname):
    self.firstname = fname
    self.lastname = lname

  def printname(self):
    print(self.firstname, self.lastname)

class Student(Person):
  def __init__(self, fname, lname):
    Person.__init__(self, fname, lname)

def test(name: str):
  print(f"hello {name}")
  print(path.join("./folder/", "filename.txt"))
  return "hello, " + name

x = Student("Mike", "Olsen")
x.printname()

print(test("billy"))