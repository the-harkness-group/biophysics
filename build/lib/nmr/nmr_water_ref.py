#!/usr/bin/env python3

def water_tsp_shifts():

    temperatures = [5,10,15,20,25,30,35,40,45,50,55,60,65,70] # Celsius
    water_shifts = [4.963959,4.916201,4.868442,4.820684,4.772925,4.725167, # Water ppm from TSP, wall chart in MSB
    4.677408,4.629650,4.581891,4.534133,4.486374,4.438616,4.390857,4.343099]

    ref_shifts = {'Temperatures':temperatures,'Shifts':water_shifts}

    return ref_shifts