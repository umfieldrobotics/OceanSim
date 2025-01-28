import numpy as np
import matplotlib.pyplot as plt

class Underwater_render():
    def __init__(self):
        # TODO 
        # Now we have the wideband_veiling_light, assume three attenuation coefficients for RGB channel and perform color correction

        self._backscatter_atte = np.array([0.1, 0.1, 0.1])
        self._direct_atte = np.array([0.1, 0.1, 0.1])


        self._depth = 0.01        # Current altitude depth measurement
        self._irradiance_0 = 1.0   # Irradiance (E) at the surface
        self._K = 0.1   #  Camera image exposure and camera pixel geometry
        # Sony_IMX117_Camera_Response       
        # [wavelength  R  G  B]
        self._camera_response = np.array([
            [400, 0.1010273973, 0.0804794521, 0.4589041096],
            [450, 0.0239726027, 0.0616438356, 0.7876712329],
            [500, 0.0445205479, 0.926369863, 0.4897260274],
            [550, 0.0479452055, 0.948630137, 0.0787671233],
            [600, 0.9366438356, 0.4914383562, 0.0342465753],
            [650, 0.7962328767, 0.154109589, 0.051369863],
            [700, 0.6386986301, 0.2910958904, 0.0856164384]
            ]) 
        self._wavelenths = self._camera_response[:,0]
        self._wavelengths_sub = (self._wavelenths[-1] - self._wavelenths[0]) / (self._wavelenths.shape[0] - 1) / 2
        self._camera_info = np.column_stack((self._camera_response[:,3], self._camera_response[:,2], self._camera_response[:,1]))
        # Jerlov I      
        # [wavelength   K_d	  b_abs	   b_sca]
        self._Jerlov_water_types = np.array([
            [400, 0.028, 0.022, 0.0062],
            [425, 0.022, 0.017, 0.00482],
            [450, 0.022, 0.018, 0.00381],
            [475, 0.021, 0.019, 0.00306],
            [500, 0.029, 0.026, 0.00249],
            [525, 0.049, 0.046, 0.00205],
            [550, 0.065, 0.062, 0.0017],
            [575, 0.085, 0.082, 0.00143],
            [600, 0.233, 0.228, 0.00122],
            [625, 0.302, 0.295, 0.00104],
            [650, 0.341, 0.334, 0.000899],
            [675, 0.444, 0.434, 0.000782],
            [700, 0.595, 0.582, 0.00685]
            ])
        self._K_d = []
        self._b_abs = []
        self._b_sca = []
        for i in range(self._Jerlov_water_types.shape[0]):
            if self._Jerlov_water_types[i,0] % 50 ==0:
                self._K_d.append(self._Jerlov_water_types[i,1])
                self._b_abs.append(self._Jerlov_water_types[i,2])
                self._b_sca.append(self._Jerlov_water_types[i,3])

        self._K_d = np.array(self._K_d)
        self._b_abs = np.array(self._b_abs)
        self._b_sca = np.array(self._b_sca)

        self._b_att = self._b_abs + self._b_sca
        
        self._irradiance = self._irradiance_0 * np.exp(-self._K_d * self._depth)
        self._veiling_light = (self._b_sca * self._irradiance / self._b_att)

        
    def calc_wideband_veiling_light(self) -> np.ndarray: 
        self.wideband_veiling_light = np.zeros(3)
        for i in range(self._wavelenths.shape[0]):
            temp_cur = self._b_sca[i] * self._irradiance[i] / self._b_att[i]
            self.wideband_veiling_light += 2.0 * self._camera_info[i,:] * temp_cur
        self.wideband_veiling_light *= 1/self._K * self._wavelengths_sub

        return self.wideband_veiling_light
    

    def render(self, raw_image):
        rendered_image = raw_image.copy()
        for i in range(raw_image.shape[0]):
            for j in range(raw_image.shape[1]):
                rendered_image[i,j,:3] = raw_image[i,j,:3] * np.exp(-self._direct_atte * self._depth) + self.wideband_veiling_light * (1 - np.exp(self._backscatter_atte * self._depth))
                rendered_image[i,j,3] = raw_image[i,j,3]
        return rendered_image
    
    def cal_backscatter(self):
        pass
        # self.backscatter = self.wideband_veiling_light * (1 - np.exp())

        


if __name__ =="__main__":


    render = Underwater_render()
    raw_image = plt.imread('rgb_1.png')
    print(f'Wideband_veiling_light:{render.calc_wideband_veiling_light()}')
    rendered_image = render.render(raw_image)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(raw_image)
    ax1.set_title('raw image')
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(rendered_image)
    ax2.set_title('rendered image')
    plt.show()