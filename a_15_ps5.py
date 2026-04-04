import pygame

# Initialize the joystick
def get_data_from_joyStick():
    pygame.init()
    pygame.joystick.init()

    joystick = pygame.joystick.Joystick(0)  # Use the first joystick
    joystick.init()

    # Used to record the last axis values, initialized to None
    last_axis_left_x, last_axis_left_y = None, None
    last_axis_right_x, last_axis_right_y = None, None
    last_axis_L2, last_axis_R2 = None, None
    last_hat = None

    # Define dead zone, inputs below this threshold are considered invalid
    dead_zone = 0.1

    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

            # Detect button press
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    print("Cross (Button 0) pressed")
                elif event.button == 1:
                    print("Circle (Button 1) pressed")
                elif event.button == 2:
                    print("Triangle (Button 2) pressed")
                elif event.button == 3:
                    print("Square (Button 3) pressed")
                elif event.button == 4:
                    print("L1 (Button 4) pressed")
                elif event.button == 5:
                    print("R1 (Button 5) pressed")
                elif event.button == 6:
                    print("L2 (Button 6) pressed")
                elif event.button == 7:
                    print("R2 (Button 7) pressed")
                elif event.button == 8:
                    print("Share (Button 8) pressed")
                elif event.button == 9:
                    print("Options (Button 9) pressed")
                elif event.button == 10:
                    print("PS button pressed")
                elif event.button == 11:
                    print("Left stick pressed")
                elif event.button == 12:
                    print("Right stick pressed")
                else:
                    print(f"Unknown button {event.button} pressed")

            # Detect button release
            if event.type == pygame.JOYBUTTONUP:
                if event.button == 0:
                    print("Cross (Button 0) released")
                elif event.button == 1:
                    print("Circle (Button 1) released")
                elif event.button == 2:
                    print("Triangle (Button 2) released")
                elif event.button == 3:
                    print("Square (Button 3) released")
                elif event.button == 4:
                    print("L1 (Button 4) released")
                elif event.button == 5:
                    print("R1 (Button 5) released")
                elif event.button == 6:
                    print("L2 (Button 6) released")
                elif event.button == 7:
                    print("R2 (Button 7) released")
                elif event.button == 8:
                    print("Share (Button 8) released")
                elif event.button == 9:
                    print("Options (Button 9) released")
                elif event.button == 10:
                    print("PS button released")
                elif event.button == 11:
                    print("Left stick released")
                elif event.button == 12:
                    print("Right stick released")
                else:
                    print(f"Unknown button {event.button} released")

        # Read the joystick axes (sticks and triggers) and print if values change
        axis_left_x = joystick.get_axis(0)  # Left stick horizontal axis
        axis_left_y = joystick.get_axis(1)  # Left stick vertical axis
        if abs(axis_left_x) > dead_zone or abs(axis_left_y) > dead_zone:
            if axis_left_x != last_axis_left_x or axis_left_y != last_axis_left_y:
                print(f"Left stick: X-axis = {axis_left_x}, Y-axis = {axis_left_y}")
                last_axis_left_x, last_axis_left_y = axis_left_x, axis_left_y

        # Right stick: AXIS 3 (horizontal) and AXIS 4 (vertical)
        axis_right_x = joystick.get_axis(3)  # Right stick horizontal axis
        axis_right_y = joystick.get_axis(4)  # Right stick vertical axis
        if abs(axis_right_x) > dead_zone or abs(axis_right_y) > dead_zone:
            if axis_right_x != last_axis_right_x or axis_right_y != last_axis_right_y:
                print(f"Right stick: X-axis = {axis_right_x}, Y-axis = {axis_right_y}")
                last_axis_right_x, last_axis_right_y = axis_right_x, axis_right_y

        # L2 and R2 triggers: AXIS 2 and AXIS 5
        axis_L2 = joystick.get_axis(2)  # L2 trigger (axis value)
        axis_R2 = joystick.get_axis(5)  # R2 trigger (axis value)
        if abs(axis_L2) > dead_zone or abs(axis_R2) > dead_zone:
            if axis_L2 != last_axis_L2 or axis_R2 != last_axis_R2:
                print(f"L2 trigger (axis value) = {axis_L2}, R2 trigger (axis value) = {axis_R2}")
                last_axis_L2, last_axis_R2 = axis_L2, axis_R2

        # Read D-pad (hat control)
        hat = joystick.get_hat(0)  # Read the D-pad
        if hat != last_hat:  # Check if the direction changes
            if hat != (0, 0):
                print(f"D-pad state: {hat}")
            last_hat = hat

        # Control output rate, update 20 times per second
        pygame.time.Clock().tick(20)

    pygame.quit()

# Call the joystick reading function
get_data_from_joyStick()
