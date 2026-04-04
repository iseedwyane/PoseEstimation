from ControlGripper_DH3 import SetCmd
import time
import pygame

# Initialize the gripper and set initial parameters
gripper = SetCmd()
time.sleep(1)
PositionReadvalue = gripper.PositionRead()
print("Position:", PositionReadvalue)

# Initial position, force, and angle
position = PositionReadvalue  # Initial position
position =60
force = 90     # Initial force
angle = 60     # Initial angle

gripper.Position(position)
gripper.angle(angle)
gripper.Force(force)

# Function to smoothly change the position within the range of 0-95
def update_position(direction):
    global position
    if direction == 'tighten':
        position += 1  # Tighten by 1 position
        if position > 95:  # Ensure it doesn't exceed the maximum value (95)
            position = 95
    elif direction == 'loosen':
        position -= 1  # Loosen by 1 position
        if position < 0:  # Ensure it doesn't go below the minimum value (0)
            position = 0
    gripper.Position(position)  # Update the gripper position
    print(f"Position set to: {position}")

# Function to smoothly change the force within the range of 10-90
def update_force(direction):
    global force
    if direction == 'increase':
        force += 5  # Increase force by 5 units
        if force > 90:  # Ensure it doesn't exceed the maximum value (90)
            force = 90
    elif direction == 'decrease':
        force -= 5  # Decrease force by 5 units
        if force < 10:  # Ensure it doesn't go below the minimum value (10)
            force = 10
    gripper.Force(force)  # Update the gripper force
    print(f"Force set to: {force}")

# Function to set angle based on button press
def set_angle(degrees):
    global angle
    angle = degrees
    gripper.angle(angle)
    print(f"Angle set to: {angle} degrees")

# Initialize the joystick
def get_data_from_joyStick():
    pygame.init()

    pygame.joystick.init()
    print(f"joystick init")
    joystick = pygame.joystick.Joystick(0)  # Use the first joystick
    joystick.init()

    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

            # Detect button press for angle control
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 2:  # Triangle button (Button 2) sets angle to 60 degrees
                    set_angle(60)
                elif event.button == 3:  # Square button (Button 3) sets angle to 0 degrees
                    set_angle(0)
                elif event.button == 1:  # Circle button (Button 1) sets angle to 90 degrees
                    set_angle(90)
                elif event.button == 0:  # Cross button (Button 0) exits the program
                    print("Cross (Button 0) pressed, exiting...")
                    done = True  # Exit the loop and quit

        # Read D-pad (hat control) for position and force control
        hat = joystick.get_hat(0)  # Read the D-pad
        if hat == (-1, 0):  # Left on D-pad (loosen position)
            print("Loosen gripper")
            for _ in range(5):  # Loosen 5 steps smoothly
                update_position('loosen')
                time.sleep(0.1)  # Smooth transition with a short delay
        elif hat == (1, 0):  # Right on D-pad (tighten position)
            print("Tighten gripper")
            for _ in range(5):  # Tighten 5 steps smoothly
                update_position('tighten')
                time.sleep(0.1)  # Smooth transition with a short delay
        elif hat == (0, 1):  # Up on D-pad (increase force)
            print("Increase gripper force")
            for _ in range(1):  # Increase force by 5
                update_force('increase')
                time.sleep(0.1)
        elif hat == (0, -1):  # Down on D-pad (decrease force)
            print("Decrease gripper force")
            for _ in range(1):  # Decrease force by 5
                update_force('decrease')
                time.sleep(0.1)

        # Control output rate, update 20 times per second
        pygame.time.Clock().tick(20)

    pygame.quit()

# Call the joystick reading function
get_data_from_joyStick()
