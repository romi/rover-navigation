import argparse
import cv2

from acre.setup import *
from acre.debug import *


def simulation_is_finished(rover, setup, i):
    return (reached_end_of_track(rover)
            or is_off_track(rover, setup)
            or reached_requested_iterations(setup, i))


def reached_end_of_track(rover):
    result = False
    if rover.x >= 44.0:
        print("Done")
        result = True
    return result
        

def is_off_track(rover, setup):
    result = False
    if setup.track.distance_from_line(rover.position) > 1.0:
        print(f"Rover out of bounds ({rover.x:0.3f}, {rover.y:0.3f})")
        result = True
    return result


def reached_requested_iterations(setup, i):
    result = False
    if setup.iterations > 0 and i == setup.iterations:
        print("Reached requested number of iterations")
        result = True
    return result
        

def main():
    # Setup is an object factory that instantiates the rover's
    # components based on the command line options.
    setup = Setup()
    rover = setup.rover
    sensor = setup.sensor
    filter = setup.filter
    controller = setup.navigation_controller
    steering = setup.steering
    track = setup.track

    assert sensor.data_type == filter.input_type
    assert filter.output_type == DataType.TRACK_ERROR
    
    map_image = cv2.imread("acre.png")
    map_height, map_width, _ = map_image.shape
    pixels_per_meter = map_width / setup.FIELD_WIDTH
    cross_track_error = 0
    orientation = 0
    radius = None

    if setup.no_map is False:  # double negation...
        draw_map(map_image, rover, pixels_per_meter,
                 cross_track_error, orientation, radius)

    print(f"rover ({rover.x:0.3f}, {rover.y:0.3f}, {np.degrees(rover.orientation):0.3f})")

    estimated_errors = None
    real_errors = None
    
    i = 0
    while True:
        debug_set_iteration(i)
        start_position = np.copy(rover.position)

        # Tell rover to move forward
        steering.drive(setup.speed, radius, setup.delta_time)

        # Predict the rover's position
        filter.predict(setup.speed, radius, setup.delta_time)
        
        # Use the sensor to measure the position
        measurement, std = sensor.measure()

        # Update the filter with the measurement
        filter.update(measurement, std)

        # Estimate position
        track_error = filter.estimate()

        # Compute the correction for the steering
        radius = controller.compute_steering(track_error)

        real_error = [track.offset(rover.position[0:2]), rover.orientation]
        if estimated_errors is None:
            estimated_errors = np.array([track_error])
            real_errors = np.array([real_error])
        else:
            estimated_errors = np.append(estimated_errors, [track_error], axis=0)
            real_errors = np.append(real_errors, [real_error], axis=0)

        if setup.no_map is False:  # double negation...
            draw_map(map_image, rover, pixels_per_meter,
                     track_error[0], track_error[1], radius)

        if radius is None:
            R = "-"
        else:
            R = f"{radius:0.3f}"
            
        print(f"[{i:04d}] Start pos:({start_position[0]:0.3f},{start_position[1]:0.3f},{np.degrees(start_position[2]):0.3f}), "
              #+ f"~Pos:({sensor._position[0]:0.3f},"
              #+ f"{sensor._position[1]:0.3f},"
              #+ f"{np.degrees(sensor._position[2]):0.3f}), "
              + f"CTE:({real_error[0]:0.3f},{np.degrees(real_error[1]):0.3f}), "
              + f"~CTE:({track_error[0]:0.3f},{np.degrees(track_error[1]):0.3f}), "
              + f"R:{R}, "
              + f"End Pos:({rover.x:0.3f},{rover.y:0.3f},{np.degrees(rover.orientation):0.3f})")
        i += 1

        if simulation_is_finished(rover, setup, i):
            break

    mean = np.mean(real_errors, axis=0)
    std = np.std(real_errors, axis=0)
    print("Real error: mean: ", mean, "std: ", std, "max: ", np.max(real_errors))
    
    mean = np.mean(estimated_errors, axis=0)
    std = np.std(estimated_errors, axis=0)
    print("Estimated error: mean: ", mean, "std: ", std)

    delta = estimated_errors - real_errors
    mean = np.mean(delta, axis=0)
    std = np.std(delta, axis=0)
    print("Error error: mean: ", mean, "std: ", std)
    
if __name__ == "__main__":
    main()
