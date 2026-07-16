# PSD definition

PSD descriptors use FFT power averaged in 24 radial bins with the DC bin excluded by edges starting at radius 1. Low/mid/high band fractions are thirds of radial PSD power normalized by total radial power. PSD slope is a linear fit of log(power) versus log(radial frequency), matching afm_descriptors.py.
