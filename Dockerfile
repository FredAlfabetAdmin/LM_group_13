FROM ros:noetic

EXPOSE 45100
EXPOSE 45101

RUN apt-get update -y 
RUN apt-get install -y python3 python3-pip git
RUN apt-get install ffmpeg libsm6 libxext6 ros-noetic-opencv-apps -y
RUN apt-get install -y firefox x11vnc xvfb
RUN echo "exec firefox" > ~/.xinitrc && chmod +x ~/.xinitrc
CMD ["v11vnc", "-create", "-forever"]
# RUN apt-get install xcb build-essential libgl1-mesa-dev libxkbcommon-x11-0 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0 libxcb-icccm4 xauth -y 
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY ./CoppeliaSim_Edu_V4_6_0_rev16_Ubuntu20_04.tar.xz ./CoppeliaSim_Edu_V4_6_0_rev16_Ubuntu20_04.tar.xz
RUN tar -xf CoppeliaSim_Edu_V4_6_0_rev16_Ubuntu20_04.tar.xz
RUN mv CoppeliaSim_Edu_V4_6_0_rev16_Ubuntu20_04 /root/coppeliasim

COPY ./requirements.txt /requirements.txt
RUN python3 -m pip install -r /requirements.txt && rm /requirements.txt

COPY ./scripts/entrypoint.bash /root/entrypoint.bash
COPY ./scripts/setup.bash /root/setup.bash
COPY ./scripts/convert_line_endings.py /root/convert_line_endings.py
WORKDIR /root/
RUN sed -i 's/\r$//' *.bash;
RUN chmod -R u+x /root/entrypoint.bash

ENTRYPOINT ["/root/entrypoint.bash"]
