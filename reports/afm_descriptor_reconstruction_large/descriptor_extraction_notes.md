# Large AFM Descriptor Extraction Notes

Descriptors were extracted from already plane-corrected ZSensor height maps. No fitted-plane subtraction was applied. Gradient, frequency, autocorrelation, and component-area descriptors use physical pixel sizes from metadata when available. Images were robust-clipped and resized to 64x64 network inputs, while scan size and pixel scale remain explicit descriptor features.
