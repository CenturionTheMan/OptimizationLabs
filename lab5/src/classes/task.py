
class Task:
    def __init__(self, p, w, d):
        self.time = p #czas
        self.weight = w #waga
        self.deadline = d #deadline

    def __repr__(self):
        return f"Task(time={self.time}, weight={self.weight}, deadline={self.deadline})"
    
    def __str__(self):
        return f"Task(time={self.time}, weight={self.weight}, deadline={self.deadline})"