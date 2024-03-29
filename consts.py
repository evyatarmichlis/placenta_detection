import os

import cv2


class Consts:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    circle_color = (0, 255, 0)
    circle_radius = 230
    circle_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 0, 0)  # Blue color in BGR
    font_thickness = 1
    TOKEN = "6864082316:AAFTOl_2EHb4jTHbdUxeQKgOgDQ93Y3Sk9M"
    chat_id = "298121019"



class Folders:
    #the comment fils are evyater michlis gmail drive
    mask_folder = "1R3UEYqlfbHg7MnwUBBwiHTlc5IL39lg1" #"1Uo8z9TeAFZRQm4j32XNTCidtQMnQMyFQ"
    placenta_main_folder = "1CVy5oqY6_TRun2SRWSrZVbPrSzqFdbUg" #"1ME5MT339M7DedPWIgiP4rK_YCaA8NETR"
    color_folder = "1VcpHEoyG4qtpFxsZtduNqmFZgjkv96Ec"  #"1sSSJWyzFxBNrJFuGtWnAS9HMLfYfQ4QG"
    depth_folder = "15EhbVBLzMjX728IG1MCFeMj1QXfmxkfI"  #"1m6t_Iknz8YAhAeTFAJdDhYCYGKeMSRhF"
    depth_csv_folder = "1cpY3EsuyKwTvJHKlssngsnmqbUqCkQTV" #1s-nysmekj_1g15olCmplF1jbc71HvSJ2"
