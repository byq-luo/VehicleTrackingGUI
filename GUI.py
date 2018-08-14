import os
import tkinter as tk
from datetime import datetime
from multiprocessing import Queue
from time import time as timer
from tkinter import ttk, filedialog

import cv2
import matplotlib
import numpy as np
import seaborn as sns
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import MCMC
import run
from ArgumentsHandler import argHandler

global q
q = Queue()
matplotlib.use('TkAgg')


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    # return the resized image
    return resized


class DeepSense_Traffic_Management(tk.Frame):

    def __init__(self, root):

        tk.Frame.__init__(self, root)
        self.overlay = None
        self.root = root
        self.root.title('DeepSense Traffic Management')
        self.root.geometry('1920x1080')
        self.root.iconbitmap('bosch.ico')
        self.pack(fill='both', expand=True)

        """
        Code the display background for the software. Has bugs, need to correct. Hence commented
        """
        # self.background = Image.open(
        #     'images/background.jpg')
        # self.background_photo = ImageTk.PhotoImage(self.background.resize((1920, 1080)))
        # self.background_label = tk.Label(root, image=self.background_photo)
        # self.background_label.place(x=0, y=0, width=1920, height=1080)

        """
        Various flags that control the gui working. Flags regarding the algorithms can be found in the run.py file
        """
        self.stop_video = False
        self.elapsed = 0
        self.draw = False
        self.drawn_line = None
        self.confirm_line = False

        """
        Customizable Flags
        """
        self.ICON_SIZE = 40
        self.CHECKED_ICON_SIZE = 25
        self.FONT = "Times New Roman"
        self.FONT_SIZE = 16

        """
        Capturing arguments from command line
        """
        FLAGS = run.manual_seeting()
        self.FLAG = argHandler()
        self.FLAG.setDefaults()
        self.FLAG.update(FLAGS)
        self.options = self.FLAG

        """
        Calling utility method input_track to initialize Sort Algorithm
        """
        self.Tracker, self.encoder = self.input_track()
        self.source = self.input_source()

        """
        Initialize Video Canvas
        """
        self.video_canvas = tk.Canvas(self, bg='white')
        self.video_canvas.place(x=40, y=50, height=600, width=900)
        self.video_canvas.bind('<Button-1>', self.onStart)
        self.video_canvas.bind('<B1-Motion>', self.onDraw)
        self.video_canvas.bind('<ButtonRelease-1>', self.onEnd)
        self.video_canvas.bind('<Button-3>', self.confirm)

        """
        Calling utility function to initialize all icons and images
        """
        self.init_images()

        """
        Variables to keep track of the checkboxes for counting
        """
        self.track_car = False
        self.track_bus = False
        self.track_motorbike = False

        """
        Initializing Control Buttons
        """
        self.Start_button = tk.Button(self, text="Start", command=self.start, bg='Red')
        self.Start_button.place(x=160, y=20, height=25, width=100)
        self.browse_button = tk.Button(self, text='Browse File', command=self.browse_file)
        self.browse_button.place(x=280, y=20, height=25, width=100)
        self.stop_button = tk.Button(self, text='Stop', command=self.Stop)
        self.stop_button.place(x=400, y=20, height=25, width=100)
        self.replay_button = tk.Button(self, text='Replay', command=self.replay)
        self.replay_button.place(x=520, y=20, height=25, width=100)
        self.graph_button = tk.Button(self, text='Graph', command=self.graph, state='disabled')
        self.graph_button.place(x=640, y=20, height=25, width=100)

        """
        Calling utility function to initialize the data table
        """
        ttk.Label(self, compound=tk.TOP, text="Data Table", font=("Times New Roman", self.FONT_SIZE)).place(x=450,
                                                                                                            y=670)
        self.create_table()

        """
        The Graph generation part
        """
        self.graph1 = ttk.Label(self, compound=tk.TOP, text="Data is being generated", image=self.graph_photo,
                                font=(self.FONT, self.FONT_SIZE))
        self.graph1.place(x=1200, y=120, height=200, width=300)
        self.graph2 = ttk.Label(self, compound=tk.TOP, text='Data is being generated', image=self.graph_photo,
                                font=(self.FONT, self.FONT_SIZE))
        self.graph2.place(x=1200, y=520, height=200, width=300)
        self.label = ttk.Label(self, text=' **Average flow is calculated per 20 sec',
                               font=(self.FONT, round(self.FONT_SIZE / 2) + 4),
                               justify=tk.RIGHT)
        self.label.place(x=1200, y=750, height=25, width=300)
        self.DEEP_label = ttk.Label(self, text='DeepSense', font=(self.FONT, self.FONT_SIZE), image=self.bosch_photo,
                                    compound=tk.TOP)
        self.DEEP_label.place(x=1200, y=20, height=100, width=300)

        self.pbar = ttk.Progressbar(self, orient=tk.HORIZONTAL, mode='determinate', length=200)
        self.pbar.place(x=40, y=1050)

        """
        This one line initializes the Neural Network for predictions
        """
        self.tfnet = run.Initialize(self.options)

        self.out = None

    def init_images(self):
        self.car_icon = Image.open('C:/Users/GRH1COB/Desktop/smartcity/Smartcity/tracking/images/car_clip.png')
        self.car_photo = ImageTk.PhotoImage(self.car_icon.resize((self.ICON_SIZE, self.ICON_SIZE)))

        self.car_photo_checked = ImageTk.PhotoImage(
            self.car_icon.resize((self.CHECKED_ICON_SIZE, self.CHECKED_ICON_SIZE)))

        self.bus_icon = Image.open('C:/Users/GRH1COB/Desktop/smartcity/Smartcity/tracking/images/bus_clip.png')
        self.bus_photo = ImageTk.PhotoImage(self.bus_icon.resize((self.ICON_SIZE, self.ICON_SIZE)))

        self.bus_photo_checked = ImageTk.PhotoImage(
            self.bus_icon.resize((self.CHECKED_ICON_SIZE, self.CHECKED_ICON_SIZE)))

        self.motorbike_icon = Image.open(
            'C:/Users/GRH1COB/Desktop/smartcity/Smartcity/tracking/images/motorbike_clip.png')
        self.motorbike_photo = ImageTk.PhotoImage(self.motorbike_icon.resize((self.ICON_SIZE, self.ICON_SIZE)))

        self.motorbike_photo_checked = ImageTk.PhotoImage(
            self.motorbike_icon.resize((self.CHECKED_ICON_SIZE, self.CHECKED_ICON_SIZE)))

        self.truck_icon = Image.open('C:/Users/GRH1COB/Desktop/smartcity/Smartcity/tracking/images/truck.png')
        self.truck_photo = ImageTk.PhotoImage(self.truck_icon.resize((self.ICON_SIZE, self.ICON_SIZE)))
        self.three_wheeler_icon = Image.open(
            'C:/Users/GRH1COB/Desktop/smartcity/Smartcity/tracking/images/three_wheeler.png')
        self.three_wheeler_photo = ImageTk.PhotoImage(self.three_wheeler_icon.resize((self.ICON_SIZE, self.ICON_SIZE)))
        self.tractor_icon = Image.open('C:/Users/GRH1COB/Desktop/smartcity/Smartcity/tracking/images/tractor.png')

        self.tractor_photo = ImageTk.PhotoImage(self.tractor_icon.resize((self.ICON_SIZE, self.ICON_SIZE)))
        self.bosch_icon = Image.open('C:/Users/GRH1COB/Desktop/smartcity/Smartcity/tracking/images/bosch.png')
        self.bosch_photo = ImageTk.PhotoImage(self.bosch_icon.resize((self.ICON_SIZE, self.ICON_SIZE)))
        self.graph_icon = Image.open('C:/Users/GRH1COB/Desktop/smartcity/Smartcity/tracking/images/graph.png')
        self.graph_photo = ImageTk.PhotoImage(self.graph_icon.resize((150, 150)))

    def create_table(self):
        """
        Utility function to create the Data Table
        :return: None
        """
        """
        Building the Data table. Grid manager is used
        """
        self.Grid = tk.Frame(self)
        self.Grid.place(x=200, y=700)
        """
         The First Column Frame containing the description for the various rows
        """
        tk.Label(self.Grid).grid(row=0, column=0)
        tk.Label(self.Grid).grid(row=1, column=0)
        self.add_table_cell(2, 0, "Count (Y/N)")
        self.add_table_cell(3, 0, "Toll Required (Y/N)")
        self.add_table_cell(4, 0, "In Flow")
        self.add_table_cell(5, 0, "Out Flow")
        self.add_table_cell(6, 0, "Avg. In Flow")
        self.add_table_cell(7, 0, "Avg. Out Flow")

        """
        The Second column containing details regarding Car
        """
        self.add_table_cell(0, 1, image=self.car_photo)
        self.add_table_cell(1, 1, text="Car")
        self.add_table_cell(2, 1, is_label=False, bind_function=self.track_car_change)
        self.add_table_cell(3, 1, is_label=False)
        self.car_in_flow_widget = self.add_table_cell(4, 1, text="0")
        self.car_out_flow_widget = self.add_table_cell(5, 1, text="0")
        self.car_avg_in_flow_widget = self.add_table_cell(6, 1, text="0")
        self.car_avg_out_flow_widget = self.add_table_cell(7, 1, text="0")

        """
        Third column containing the details regarding Bus
        """
        self.add_table_cell(0, 2, image=self.bus_photo)
        self.add_table_cell(1, 2, text="Bus")
        self.add_table_cell(2, 2, is_label=False, bind_function=self.track_bus_change)
        self.add_table_cell(3, 2, is_label=False)
        self.bus_in_flow_widget = self.add_table_cell(4, 2, text="0")
        self.bus_out_flow_widget = self.add_table_cell(5, 2, text="0")
        self.bus_avg_in_flow_widget = self.add_table_cell(6, 2, text="0")
        self.bus_avg_out_flow_widget = self.add_table_cell(7, 2, text="0")

        """
        Fourth column containing the details regarding Motorbike
        """
        self.add_table_cell(0, 3, image=self.motorbike_photo)
        self.add_table_cell(1, 3, text="Motorbike")
        self.add_table_cell(2, 3, is_label=False, bind_function=self.track_motorbike_change)
        self.add_table_cell(3, 3, is_label=False)
        self.motorbike_in_flow_widget = self.add_table_cell(4, 3, text="0")
        self.motorbike_out_flow_widget = self.add_table_cell(5, 3, text="0")
        self.motorbike_avg_in_flow_widget = self.add_table_cell(6, 3, text="0")
        self.motorbike_avg_out_flow_widget = self.add_table_cell(7, 3, text="0")

        """
        Fifth column containing the details regarding Truck
        """
        self.add_table_cell(0, 4, image=self.truck_photo)
        self.add_table_cell(1, 4, text="Truck")
        self.add_table_cell(2, 4, is_label=False, bind_function=None)
        self.add_table_cell(3, 4, is_label=False)
        self.truck_in_flow_widget = self.add_table_cell(4, 4, text="0")
        self.truck_out_flow_widget = self.add_table_cell(5, 4, text="0")
        self.truck_avg_in_flow_widget = self.add_table_cell(6, 4, text="0")
        self.truck_avg_out_flow_widget = self.add_table_cell(7, 4, text="0")

        """
        Sixth column containing the details regarding Three Wheeled Vehicles
        """

        self.add_table_cell(0, 5, image=self.three_wheeler_photo)
        self.add_table_cell(1, 5, text="Three Wheeler")
        self.add_table_cell(2, 5, is_label=False, bind_function=None)
        self.add_table_cell(3, 5, is_label=False)
        self.truck_in_flow_widget = self.add_table_cell(4, 5, text="0")
        self.truck_out_flow_widget = self.add_table_cell(5, 5, text="0")
        self.truck_avg_in_flow_widget = self.add_table_cell(6, 5, text="0")
        self.truck_avg_out_flow_widget = self.add_table_cell(7, 5, text="0")

        """
        Seventh column containing the details regarding Tractor
        """
        self.add_table_cell(0, 6, image=self.tractor_photo)
        self.add_table_cell(1, 6, text="Tractor")
        self.add_table_cell(2, 6, is_label=False, bind_function=None)
        self.add_table_cell(3, 6, is_label=False)
        self.truck_in_flow_widget = self.add_table_cell(4, 6, text="0")
        self.truck_out_flow_widget = self.add_table_cell(5, 6, text="0")
        self.truck_avg_in_flow_widget = self.add_table_cell(6, 6, text="0")
        self.truck_avg_out_flow_widget = self.add_table_cell(7, 6, text="0")

    def add_table_cell(self, row, column, text=None, image=None, is_label=True, bind_function=None):
        """
        Utility function to add a cell to the table. If both text and image are left blank, a blank cell is created and added
        :param row: row of the grid
        :param column: column of the grid
        :param text: the text to be displayed. Leave blank if you want none.
        :param image: the image to be displayed. Leave blank if you want none.
        :param is_label: To indicate weather to add a label or a checkbox
        :param bind_function: The on click function if a checkbox is needed
        :return: the widget cell that is created and added to the grid table is returned
        """
        if is_label:
            temp = tk.Label(self.Grid, text=text, image=image, borderwidth=2, relief="ridge")
            temp.grid(row=row, column=column, sticky=(tk.N, tk.S, tk.E, tk.W), ipadx=5, ipady=5)
            for i in range(1, 4):
                temp.columnconfigure(i, weight=1)
            temp.rowconfigure(row, weight=1)
            temp.bind("<Button-1>", bind_function)
        else:
            temp = tk.Checkbutton(self.Grid, text=text, image=image, borderwidth=2, relief="ridge")
            temp.grid(row=row, column=column, sticky=(tk.N, tk.S, tk.E, tk.W), ipadx=5, ipady=5)
            for i in range(1, 4):
                temp.columnconfigure(i, weight=1)
            temp.rowconfigure(row, weight=1)
            temp.bind("<Button-1>", bind_function)
        return temp

    def track_car_change(self, event):
        self.track_car = not self.track_car

    def track_bus_change(self, event):
        self.track_bus = not self.track_bus

    def track_motorbike_change(self, event):
        self.track_motorbike = not self.track_motorbike

    def start(self):

        self.Start_button.config(bg='Green')
        self.stop_video = False

        self.update()

    def browse_file(self):
        if self.stop_video:
            self.source.release()
            self.confirm_line = False
            filename = filedialog.askopenfilename()
            self.options.demo = filename
            self.source = self.input_source()
            # self.Tracker.frame_width = self.frame_width
            # self.Tracker.frame_height = self.frame_height
            self.Tracker.new_video = True
            self.elapsed = 0
            self.Tracker.frame_count = 0

    def Stop(self):
        self.stop_video = True
        self.Start_button.config(bg='Red')

    def replay(self):
        if self.stop_video:
            self.source.release()

            self.confirm_line = False
            self.source = self.input_source()
            self.Tracker.new_video = True
            self.elapsed = 0
            self.Tracker.frame_count = 0
            self.start()

    def onStart(self, event):
        if not self.confirm_line:
            self.start_x = event.x
            self.start_y = event.y
            self.draw = True
            self.end = False
            event.widget.delete(self.drawn_line)

    def graph(self):

        self.Posterior_in = MCMC.sampler(self.data_2min_in, samples=15000, mu_init=1.5)

        self.Posterior_out = MCMC.sampler(self.data_2min_out, samples=15000, mu_init=1.5)

        self.graph_call()

    def graph_call(self):
        # self.Posterior = MCMC.sampler(self.data_2min, samples=5000, mu_init=1.5)

        print('I am in the graph')
        f_in = Figure(figsize=(4, 3.5), dpi=80)
        f_out = Figure(figsize=(4, 3.5), dpi=80)
        f_in.suptitle("Predicted Traffic IN FLOW")
        f_out.suptitle("Predicted Traffic OUT FLOW")
        # if self.ax!=None:
        #
        #     self.ax.clear()
        #     self.bx.clear()
        self.ax = f_in.add_subplot(111)
        self.bx = f_out.add_subplot(111)

        x = np.linspace(1, 2.5, 800)
        sns.distplot(self.Posterior_in[500:], ax=self.ax, label='estimated posterior')
        sns.distplot(self.Posterior_out[500:], ax=self.bx)
        # post_in = MCMC.calc_posterior_analytical(self.data_2min_in,x,1.6,1)

        # self.ax.plot(x,post,'g', Label = 'analytic posterior')
        _ = self.ax.set(xlabel='AVERAGE', ylabel='BELIEF')
        _ = self.bx.set(xlabel='AVERAGE', ylabel='BELIEF')

        canvas_in = FigureCanvasTkAgg(f_in, self)
        canvas_out = FigureCanvasTkAgg(f_out, self)
        canvas_in.draw()
        canvas_out.draw()
        canvas_in.get_tk_widget().place(x=1080, y=120)
        canvas_out.get_tk_widget().place(x=1080, y=450)
        self.graph_button.config(state='disabled')

    def onDraw(self, event):
        self.line_canvas = event.widget
        if self.draw and self.end == False:
            self.line_canvas.delete(self.drawn_line)
            self.drawn_line = self.line_canvas.create_line(self.start_x, self.start_y, event.x, event.y, fill='red')

    def onEnd(self, event):
        if self.draw:
            self.end_x = event.x
            self.end_y = event.y
            self.end = True
            self.draw = None

    def confirm(self, event):
        self.start = timer()
        self.confirm_line = True
        self.till_last_5sec_in = 0
        self.till_last_5sec_out = 0
        self.data_5sec_in = []
        self.data_5sec_out = []
        self.ax = None
        # self.line_coordinate_video = self.conversion()
        self.Tracker.line_coordinate = self.conversion()

    def conversion(self):
        start_x = round(self.start_x * (self.frame_width / 900))
        end_x = round(self.end_x * (self.frame_width / 900))
        start_y = round(self.start_y * (self.frame_height / 600))
        end_y = round(self.end_y * (self.frame_height / 600))
        line_coordinate = [start_x, start_y, end_x, end_y]
        return line_coordinate

    def input_source(self):
        file = self.options.demo

        if file == 'camera':

            file = 0
            self.frame_rate = 1
            camera = cv2.VideoCapture(file)



        else:
            assert os.path.isfile(file), \
                'file {} does not exist'.format(file)

            camera = cv2.VideoCapture(file)
            self.frame_rate = camera.get(cv2.CAP_PROP_FPS)

            self.frame_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.frame_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.Tracker.frame_width = self.frame_width
            self.Tracker.frame_height = self.frame_height
            self.Tracker.frame_rate = self.frame_rate
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            # date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            file = file.split("/")[-1].split(".")[0]
            # LOCATION = "\\\\BOSCH.COM\\DfsRB\\DfsIN\\LOC\\Cob\\NE1\\Function\\Aerospace_Deep_Sense\\02_Development\\05_Exchange_Folder\\08_Srinath\\Processed Video 9th August\\Toll Booth\\{}.mp4 ".format(file)
            OUTPUT_FILE_NAME = 'C:\\Users\\GRH1COB\\Desktop\\smartcity\\Smartcity\\tracking\\output_video\\{}.mp4'.format(file)
            # OUTPUT_FILE_NAME = LOCATION
            VIDEO_SCALE_RATIO = 0.5
            RATIO_OF_BELOW_BOX = 0.35
            _, frame = camera.read()
            frame = cv2.resize(frame, None, fx=VIDEO_SCALE_RATIO, fy=VIDEO_SCALE_RATIO,
                               interpolation=cv2.INTER_LINEAR)
            width = frame.shape[1]
            height = frame.shape[0]
            b_height = round(frame.shape[0] * RATIO_OF_BELOW_BOX)

            blank_image = np.zeros((b_height, width, 3), np.uint8)
            blank_image[np.where((blank_image == [0, 0, 0]).all(axis=2))] = [240, 240, 240]
            img = np.row_stack((frame, blank_image))
            fheight = img.shape[0]
            fwidth = img.shape[1]
            self.out = cv2.VideoWriter(OUTPUT_FILE_NAME, fourcc, self.frame_rate, (fwidth, fheight), True)

        return camera

    def raw_video(self):
        if self.source.isOpened:
            _, frame = self.source.read()
        return frame

    def input_track(self):
        """
        Utility function to initialize the sort algorithm
        :return: None
        """
        if self.options.tracker == 'sort':
            from sort.sort import Sort
            encoder = None
            Tracker = Sort()

        return Tracker, encoder

    def get_postprocessed(self):
        frame = self.raw_video()

        if frame is None:
            return None
        if self.Tracker.frame_count >= 1:
            self.Tracker.new_video = False

        preprocessed = self.tfnet.framework.preprocess(frame)

        buffer_inp = list()
        buffer_pre = list()
        buffer_inp.append(frame)
        buffer_pre.append(preprocessed)

        feed_dict = {self.tfnet.inp: buffer_pre}
        net_out = self.tfnet.sess.run(self.tfnet.out, feed_dict)
        for img, single_out in zip(buffer_inp, net_out):
            postprocessed = self.tfnet.framework.postprocess(single_out, img,
                                                             frame_id=self.elapsed, csv_file=None,
                                                             csv=None, mask=None, encoder=self.encoder,
                                                             tracker=self.Tracker)

        return postprocessed

    def update(self):
        """
        This method is called for every frame by Tkinter. All the frame processing goes here.
        :return: None
        """
        if not self.confirm_line:
            self.current_frame = self.raw_video()
        else:

            self.current_frame = self.get_postprocessed()

        if not self.stop_video:
            if self.current_frame is not None:
                self.cv2_image = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGBA)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.cv2_image).resize((900, 600)))
                self.video_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                # self.car_passed.create_text(fill="darkblue",font="Times 20 italic bold",
                """
                Customizations for the Saved video
                """
                FONT = cv2.FONT_HERSHEY_SIMPLEX
                FONT_SCALE = 0.4
                FONT_SCALE_HEADING = 0.6
                FONT_COLOR = (0, 0, 0)
                VIDEO_SCALE_RATIO = 0.5
                RATIO_OF_BELOW_BOX = 0.35

                frame = cv2.resize(self.current_frame, None, fx=VIDEO_SCALE_RATIO, fy=VIDEO_SCALE_RATIO,
                                   interpolation=cv2.INTER_LINEAR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                width = frame.shape[1]
                height = frame.shape[0]
                img_path = 'icons/bosch.png'
                logo = cv2.imread(img_path, -1)
                watermark = image_resize(logo, height=50)
                watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)
                overlay = np.zeros((height, width, 4), dtype='uint8')
                watermark_h, watermark_w, watermark_c = watermark.shape
                for i in range(0, watermark_h):
                    for j in range(0, watermark_w):
                        if watermark[i, j][3] != 0:
                            overlay[10 + i, 10 + j] = watermark[i, j]
                cv2.addWeighted(overlay, 1, frame, 1, 0, frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                width = frame.shape[1]
                height = round(frame.shape[0] * RATIO_OF_BELOW_BOX)
                blank_image = np.zeros((height, width, 3), np.uint8)
                blank_image[np.where((blank_image == [0, 0, 0]).all(axis=2))] = [240, 240, 240]
                cv2.putText(frame, "DeepSense", (width - int(width * 0.25), round(height * 0.2)), FONT, 1,
                            (255, 255, 255), 2)
                """
                This part of code displays the algorithm's output to the canvas
                """

                if self.confirm_line:
                    self.elapsed += 1

                    vehicle_count = self.Tracker.vehicle_count
                    if self.track_car:
                        self.car_in_flow_widget.config(text="{}".format(vehicle_count[0]))
                        self.car_out_flow_widget.config(text="{}".format(vehicle_count[4]))
                        self.car_avg_in_flow_widget.config(text="{}".format(vehicle_count[7]))
                        self.car_avg_out_flow_widget.config(text="{}".format(vehicle_count[10]))
                    else:
                        vehicle_count[0] = 0
                        vehicle_count[4] = 0
                        vehicle_count[7] = 0
                        vehicle_count[10] = 0
                    if self.track_bus:
                        self.bus_in_flow_widget.config(text="{}".format(vehicle_count[1]))
                        self.bus_out_flow_widget.config(text="{}".format(vehicle_count[5]))
                        self.bus_avg_in_flow_widget.config(text="{}".format(vehicle_count[8]))
                        self.bus_avg_out_flow_widget.config(text="{}".format(vehicle_count[11]))
                    else:
                        vehicle_count[1] = 0
                        vehicle_count[5] = 0
                        vehicle_count[8] = 0
                        vehicle_count[11] = 0
                    if self.track_motorbike:
                        self.motorbike_in_flow_widget.config(text="{}".format(vehicle_count[2]))
                        self.motorbike_out_flow_widget.config(text="{}".format(vehicle_count[6]))
                        self.motorbike_avg_in_flow_widget.config(text="{}".format(vehicle_count[9]))
                        self.motorbike_avg_out_flow_widget.config(text="{}".format(vehicle_count[12]))
                    else:
                        vehicle_count[2] = 0
                        vehicle_count[6] = 0
                        vehicle_count[9] = 0
                        vehicle_count[12] = 0

                    if self.elapsed % (5 * self.frame_rate) == 0:
                        total_current_in = vehicle_count[0] + vehicle_count[1] + vehicle_count[2]
                        total_current_out = vehicle_count[4] + vehicle_count[5] + vehicle_count[6]
                        bet_5sec_in = total_current_in - self.till_last_5sec_in
                        bet_5sec_out = total_current_out - self.till_last_5sec_out
                        # print(self.elapsed / (timer() - self.start))
                        self.data_5sec_in.append(bet_5sec_in)
                        self.data_5sec_out.append(bet_5sec_in)

                        if self.elapsed % (120 * self.frame_rate) == 0:
                            self.graph_button.config(state='normal')
                            # print('data received')
                            self.data_2min_in = np.array(self.data_5sec_in)
                            self.data_2min_out = np.array(self.data_5sec_out)
                            self.data_5sec_in = []
                            self.data_5sec_out = []

                        self.till_last_5sec_in = total_current_in
                        self.till_last_5sec_out = total_current_out
                    """
                    Adding text to Output Video
                    """
                    # Car Data
                    cv2.putText(blank_image, 'Vehicle Type', (30, 30), FONT, FONT_SCALE_HEADING, FONT_COLOR, 1)
                    cv2.putText(blank_image, 'Count (Y/N)', (30, 50), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, 'Toll(Y/N)', (30, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, 'In Flow', (30, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, 'Out Flow', (30, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, 'Avg In Flow', (30, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, 'Avg Out Flow', (30, 150), FONT, FONT_SCALE, FONT_COLOR, 1)

                    # Car Data:
                    cv2.putText(blank_image, 'Car', (180, 30), FONT, FONT_SCALE_HEADING, FONT_COLOR, 1)
                    cv2.putText(blank_image, 'Yes' if self.track_car else 'No', (180, 50), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, 'No', (180, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '{}'.format(vehicle_count[0]), (180, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '{}'.format(vehicle_count[4]), (180, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '{}'.format(vehicle_count[7]), (180, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '{}'.format(vehicle_count[10]), (180, 150), FONT, FONT_SCALE, FONT_COLOR, 1)

                    # Bus Data:
                    cv2.putText(blank_image, 'Bus', (255, 30), FONT, FONT_SCALE_HEADING, FONT_COLOR, 1)
                    cv2.putText(blank_image, 'Yes' if self.track_bus else 'No', (255, 50), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, 'No', (255, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '{}'.format(vehicle_count[1]), (255, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '{}'.format(vehicle_count[5]), (255, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '{}'.format(vehicle_count[8]), (255, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '{}'.format(vehicle_count[11]), (255, 150), FONT, FONT_SCALE, FONT_COLOR, 1)

                    # Bike Data:
                    cv2.putText(blank_image, 'Bike', (330, 30), FONT, FONT_SCALE_HEADING, FONT_COLOR, 1)
                    cv2.putText(blank_image, 'Yes' if self.track_motorbike else 'No', (330, 50), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, 'No', (330, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '{}'.format(vehicle_count[2]), (330, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '{}'.format(vehicle_count[6]), (330, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '{}'.format(vehicle_count[9]), (330, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '{}'.format(vehicle_count[12]), (330, 150), FONT, FONT_SCALE, FONT_COLOR, 1)

                    # Truck Data:
                    cv2.putText(blank_image, 'Truck', (405, 30), FONT, FONT_SCALE_HEADING, FONT_COLOR, 1)
                    cv2.putText(blank_image, 'No', (405, 50), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, 'No', (405, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '0', (405, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '0', (405, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '0', (405, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '0', (405, 150), FONT, FONT_SCALE, FONT_COLOR, 1)

                    # Rickshaw Data:
                    cv2.putText(blank_image, 'Three Wheeler', (480, 30), FONT, FONT_SCALE_HEADING, FONT_COLOR, 1)
                    cv2.putText(blank_image, 'No', (480, 50), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, 'No', (480, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '0', (480, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '0', (480, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '0', (480, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '0', (480, 150), FONT, FONT_SCALE, FONT_COLOR, 1)

                    # Tractor Data:
                    cv2.putText(blank_image, 'Tractor', (630, 30), FONT, FONT_SCALE_HEADING, FONT_COLOR, 1)
                    cv2.putText(blank_image, 'No', (630, 50), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, 'No', (630, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '0', (630, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '0', (630, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '0', (630, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, '0', (630, 150), FONT, FONT_SCALE, FONT_COLOR, 1)

                else:
                    cv2.putText(blank_image, 'Counting Not Turned On', (10, 30), FONT, FONT_SCALE, FONT_COLOR, 1)
                if self.out:
                    img = np.row_stack((frame, blank_image))
                    self.out.write(img)
                # print(self.camera_frame_rate())

                self.root.after(20, self.update)


            else:
                self.out.release()
                self.source.release()


root = tk.Tk()

window = DeepSense_Traffic_Management(root)
root.mainloop()

# window  = DeepSense_Traffic_Management()
