# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:03:21 2020

@author: johan
"""



# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:32:34 2019
This class uses tkinter to build and handle the UI for the Proprocessing Tool
@author: johan
"""


from tkinter import Checkbutton
from tkinter import IntVar
from tkinter import Tk
from tkinter import END
from tkinter import Label
from tkinter import LEFT
from tkinter import W
from tkinter import Button
from tkinter import Entry
from tkinter import filedialog
from tkinter import messagebox
from tkinter.ttk import Progressbar
from tkinter.ttk import Notebook
from tkinter.ttk import Frame
from tkinter.ttk import Separator
import sys
import os
import analyze_video
from threading import Event
import threading
import tkinter.scrolledtext as scrolledtext
import datetime



pre_tool = None
post_tool = None

window_width = 650
window_height = 800



def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

analyze_videos_thread = None

def pause_analyze_videos(pause_event,progress_callback):
    pause_event.set()
    
    progress_callback("started_stopping_script")
    analyze_videos_thread.join()
    pause_event.clear()
    progress_callback("stopped_script")
    progress_callback("Pause")



def start_analyze_videos(videos, results_folder, visualize,pause_event, progress_callback, config_file_path):
    
    if not os.path.isdir(results_folder):
        messagebox.showerror("Error", "The Results Folder is not a valid folder. Please check the spelling.")
        return
    if videos == []:
        messagebox.showerror("Error", "There are no videos provided. Please select some videos that should be analyzed.")
        return
    for video in videos:
        if not os.path.isfile(video):
            messagebox.showerror("Error", "The following video does not exist:\n\n" + video + "\n\nPlease check the spelling")
            return
    
        
    #analyze_video.analyze_videos(constants.bee_model_path, constants.hole_model_path, videos, results_folder, visualize, progress_callback, pause_event)
    global analyze_videos_thread
    my_args=(videos, results_folder, visualize, progress_callback, pause_event,config_file_path)
    analyze_videos_thread = threading.Thread(target=analyze_video.analyze_videos, args=my_args)
    analyze_videos_thread.daemon = True
    analyze_videos_thread.start()
    progress_callback("started_script")
    progress_callback("Start")

    
    
def convert_paths_list(paths_string):
    result = []
    paths = paths_string.split(";")
    for path in paths:
        stripped_path = path.rstrip().lstrip()
        if stripped_path != "": 
            result.append(stripped_path)
    return result
        

def start_ui():
    
    main_window = Tk()
    main_window.geometry(str(window_width) + "x" + str(window_height))
    main_window.iconbitmap(resource_path('bee.ico'))
    
    
    main_window.title("Bee Movement Analyzer")
    
    tabControl = Notebook(main_window)
    
    
    
    description = "Choose a results folder and one or more input videos that should be analyzed. Please separate your input video paths with semicolons! If you enable the 'Visualize Results' option, the program will save the videos to the results folder with all bee detections visualized. Note that by enabling this option, more compute time is needed. The program will analyze all videos and save all results to the results folder. Make sure that there is enough space in the results folder, especially if you enable the 'Visualize Results' option.\n\nShould you wish to pause the program, click the 'Pause' button and wait until the program has paused all computations. This can take up to a few minutes. The computation can later be resumed. Please read the documentation PDF for further information."

    
    '''Init Tab 1'''
    tab1 = Frame(tabControl)
    tab1.columnconfigure(1, weight=1)
    
    
    #Description
    lbl = Label(tab1, justify=LEFT, wraplength=window_width-20, text=description)
    lbl.grid(column=0, row=0, pady=5, padx = 5, columnspan=4, sticky="ew")
    
    
    
    Separator(tab1).grid(column=0, row=28, pady=5, padx = 5, columnspan=4, sticky="we")
    progress_label = Label(tab1,justify=LEFT, text="Progress: ")
    progress_label.grid(column=0, row=29, pady=5, padx = 5, columnspan=1, sticky=W)

    #Progressbars
    detect_bees_progress_bar = Progressbar(tab1, length=10000)
    detect_bees_progress_bar.grid(column=1, row=30, pady=5, padx = 5, sticky=W, columnspan=3)
    detect_bees_progress_bar['value'] = 0
    detect_bees_progress_label = Label(tab1,justify=LEFT, text="Detecting bees: ")
    detect_bees_progress_label.grid(column=0, row=30, pady=5, padx = 5, columnspan=1, sticky=W)

    detect_holes_progress_bar = Progressbar(tab1, length=10000)
    detect_holes_progress_bar.grid(column=1, row=31, pady=5, padx = 5, sticky=W, columnspan=3)
    detect_holes_progress_bar['value'] = 0
    detect_holes_progress_label = Label(tab1,justify=LEFT, text="Detecting holes: ")
    detect_holes_progress_label.grid(column=0, row=31, pady=5, padx = 5, columnspan=1, sticky=W)

    detect_colors_progress_bar = Progressbar(tab1, length=10000)
    detect_colors_progress_bar.grid(column=1, row=32, pady=5, padx = 5, sticky=W, columnspan=3)
    detect_colors_progress_bar['value'] = 0
    detect_colors_progress_bar_label = Label(tab1,justify=LEFT, text="Detecting colors: ")
    detect_colors_progress_bar_label.grid(column=0, row=32, pady=5, padx = 5, columnspan=1, sticky=W)

    detect_numbers_progress_bar = Progressbar(tab1, length=10000)
    detect_numbers_progress_bar.grid(column=1, row=33, pady=5, padx = 5, sticky=W, columnspan=3)
    detect_numbers_progress_bar['value'] = 0
    detect_numbers_progress_label = Label(tab1,justify=LEFT, text="Detecting numbers: ")
    detect_numbers_progress_label.grid(column=0, row=33, pady=5, padx = 5, columnspan=1, sticky=W)

    visualize_progress_bar = Progressbar(tab1, length=10000)
    visualize_progress_bar.grid(column=1, row=34, pady=5, padx = 5, sticky=W, columnspan=3)
    visualize_progress_bar['value'] = 0
    visualize_progress_label = Label(tab1,justify=LEFT, text="Visualizing results: ")
    visualize_progress_label.grid(column=0, row=34, pady=5, padx = 5, columnspan=1, sticky=W)

    
    #Text Output:
    output_label = Label(tab1,justify=LEFT, text="Output: ")
    output_label.grid(column=0, row=49, pady=5, padx = 5, columnspan=1, sticky=W)

    console = scrolledtext.ScrolledText(tab1, undo=True, height=12)
    #console['font'] = ('consolas', '12')
    console.config(state="disabled")
    console.grid(column=0, row=50, pady=5, padx = 5, columnspan=4, sticky="ew")


    
    video_paths_input = Entry(tab1, width=200)
    video_paths_input.grid(column=1, row=10, pady=5, padx = 5, columnspan=2, sticky=W)
    
    
    
    def getInputVideoFilePaths():
        input_files = filedialog.askopenfilenames(filetypes = [("Video Files", "*.mp4")])
        if not input_files == "":
            display_string = ""
            for input_file in input_files:
                display_string += input_file + "; "
            #Update UI
            video_paths_input.delete(0,END)
            video_paths_input.insert(0,display_string)


    select_input_videos_button = Button(tab1, text="Select Input Videos",command=getInputVideoFilePaths)
    select_input_videos_button.grid(column=3, row=10, pady=5, padx = 5, sticky=W)
    input_videos_label = Label(tab1,justify=LEFT, text="Input Videos: ")
    input_videos_label.grid(column=0, row=10, pady=5, padx = 5, columnspan=1, sticky=W)
    

    output_path_input = Entry(tab1,width=200)
    output_path_input.grid(column=1, row=11, pady=5, padx = 5, columnspan=2, sticky=W)

    
    def getResultsFolder():
        output_dir = filedialog.askdirectory()
        if not output_dir == "":
            #Update UI
            output_path_input.delete(0,END)
            output_path_input.insert(0,output_dir)
    
    open_output_dir_button = Button(tab1, text="Select Results Folder",command=getResultsFolder)
    open_output_dir_button.grid(column=3, row=11, pady=5, padx = 5, sticky=W)
    open_output_label = Label(tab1,justify=LEFT, text="Results Folder: ")
    open_output_label.grid(column=0, row=11, pady=5, padx = 5, columnspan=1, sticky=W)
    
    
    config_file_input = Entry(tab1,width=200)
    config_file_input.grid(column=1, row=12, pady=5, padx = 5, columnspan=2, sticky=W)
    
    def getConfigFilePath():
        input_file = filedialog.askopenfilename(filetypes = [("Python File", "*.py")])
        if not input_file == "":
            #Update UI
            config_file_input.delete(0,END)
            config_file_input.insert(0,input_file)

    config_file_input_button = Button(tab1, text="Select Config File",command=getConfigFilePath)
    config_file_input_button.grid(column=3, row=12, pady=5, padx = 5, sticky=W)
    config_file_label = Label(tab1,justify=LEFT, text="Config File (optional): ")
    config_file_label.grid(column=0, row=12, pady=5, padx = 5, columnspan=1, sticky=W)

    
    visualize_variable = IntVar()
    check_button = Checkbutton(tab1, text="Visualize Results", variable=visualize_variable)
    check_button.grid(column=1, row=13, pady=5, padx = 5, columnspan=2, sticky=W)


    pause_event = Event()
    
    run_button = Button(tab1, text="Start")
    
    def progress_callback(progress):
        if progress == "started_script":
            run_button.configure(text="Pause",state="normal", command=lambda: threading.Thread(target=pause_analyze_videos, args=(pause_event,progress_callback,)).start())
            detect_bees_progress_bar['value'] = 0
            visualize_progress_bar['value'] = 0
            detect_holes_progress_bar['value'] = 0
            detect_numbers_progress_bar['value'] = 0
            detect_colors_progress_bar['value'] = 0
        elif progress == "started_stopping_script":
            run_button.configure(text="Please wait...", state="disabled")
        elif progress == "stopped_script":
            run_button.configure(text="Start", state="normal", command=lambda: start_analyze_videos(convert_paths_list(video_paths_input.get()),output_path_input.get(),visualize_variable.get(),pause_event,progress_callback,config_file_input.get()))
        elif progress == "Success. All videos are analyzed!":
            run_button.configure(text="Start", state="normal", command=lambda: start_analyze_videos(convert_paths_list(video_paths_input.get()),output_path_input.get(),visualize_variable.get(),pause_event,progress_callback,config_file_input.get()))
            progress_callback("Success. All videos are analyzed.")
        elif type(progress) == tuple:
            if type(progress[1]) == float:
                #it is a numerical progress
                if progress[0] == "detect_bees":
                    detect_bees_progress_bar['value'] = progress[1]*100
                elif progress[0] == "visualize":
                    visualize_progress_bar['value'] = progress[1]*100
                elif progress[0] == "detect_holes":
                    detect_holes_progress_bar['value'] = progress[1]*100
                elif progress[0] == "detect_colors":
                    detect_colors_progress_bar['value'] = progress[1]*100
                elif progress[0] == "detect_numbers":
                    detect_numbers_progress_bar['value'] = progress[1]*100
        elif type(progress) == str:
            console.configure(state='normal')
            
            dateTimeObj = datetime.datetime.now()
            timestampStr = dateTimeObj.strftime("%d.%b %H:%M:%S")
            console.insert(END,timestampStr + ": " + progress + "\n")
            console.see(END)
            
            console.config(state="disabled")
            


        
            

    
    
    run_button.configure(state="normal",command=lambda: start_analyze_videos(convert_paths_list(video_paths_input.get()),output_path_input.get(),visualize_variable.get(),pause_event,progress_callback,config_file_input.get()))
    
    run_button.grid(column=0, row=20, pady=5, padx = 5,columnspan=4, sticky='ew')
    
    
    
    
    tabControl.add(tab1, text='Analyze Bee Videos')
    
    

    
    
    tabControl.pack(expand=True, fill="both")  # Pack to make visible
        
    
    
    
    
    output_path_input.delete(0,END)
    output_path_input.insert(0,"C:/Users/johan/Downloads/tes")
    video_paths_input.delete(0,END)
    video_paths_input.insert(0,"C:/Users/johan/Downloads/Test4_4.MP4; ")


    
    
    
    
    main_window.mainloop()


if __name__== "__main__":
 
    start_ui()