from anytree import Node, RenderTree

udo = Node("Udo")
marc = Node("Marc", parent=udo)
lian = Node("Lian", parent=marc)
dan = Node("Dan", parent=udo)
jet = Node("Jet", parent=dan)
jan = Node("Jan", parent=dan)
joe = Node("Joe", parent=dan)

print(udo)
# output: Node('/Udo')
print(joe)
# output: Node('/Udo/Dan/Joe')

for pre, fill, node in RenderTree(udo):
    print("%s%s" % (pre, node.name))

print("Dan children")    
print(dan.children)
# output: (Node('/Udo/Dan/Jet'), Node('/Udo/Dan/Jan'), Node('/Udo/Dan/Joe'))

print("Jan depth")
print(jan.depth)

print("Jan siblings")
print(jan.siblings)