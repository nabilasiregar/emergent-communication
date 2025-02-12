distance to target location from nest = num of times they waggle
resource direction = waggle direction angle 

Referential communication codes information, the dancer (sender) encodes the polar coordinates of a resource relative to the nest

Sender should know:
- polar coordinate of the nest node
- food quality (but does not matter if we only have 1 food source)

Sender's message contains:
- distance
- direction

Nodes:
1. Food node
2. Nest node
3. Sender's position node
4. Receiver's position node (i think it should be the same position as the nest because hivemates live in the nest?)

Edge attribute:
- Distance (in meters), this is to calculate the total distance from nest node to food node
- The receiver then will be able to keep track how far they have traveled and stop once the total distance
has been travelled, it has to be the exact same total distance, because otherwise they travel in the wrong direction

Node attribute:
- Direction (in angles, if discrete then in 8 quadrants). This is to help the bee to decide to move up, down, right, left based on cardinal directions, so this means the maximum number of connections a node have should be 8??

Message looks like:
"['N', 'NE', 'S', 'W', 60] this means from the nest travel N -> NE -> S -> W and it should find the food source after traveling for 60m
