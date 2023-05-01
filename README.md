# Person Following

In the ros2 workspace (turtlebot4_ws in our case)

cd turtlebot4_ws/src

git clone https://github.com/e-elias/person.git

cd ..

colcon build

--In another terminal

cd turtlebot4_ws 

source install/setup.bash

ros2 run person person_following
