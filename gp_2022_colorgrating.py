import numpy as np

import vxpy.core.protocol as vxprotocol

from visuals.spherical_grating import SphericalColorContrastGrating
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground



class ProtocolGP(vxprotocol.StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticPhasicProtocol.__init__(self, *args, **kwargs)

        angular_period_degrees = 30
        angular_velocity_degrees = 30

        c = [0., 0.1, 0.18, 0.32, 0.56, 1.0]
        params = []
        for d in [-1, 1]:
            for c1 in c:
                for c2 in c:
                    params.append((c1, c2, d))
        np.random.seed(1)
        params = np.random.permutation(params)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(3):
            for c1, c2, d in params:

                phase = vxprotocol.Phase(duration=4)
                phase.set_visual(SphericalColorContrastGrating,
                                 {SphericalColorContrastGrating.angular_period: angular_period_degrees,
                                  SphericalColorContrastGrating.angular_velocity: 0.,
                                  SphericalColorContrastGrating.waveform: 'rectangular',
                                  SphericalColorContrastGrating.motion_type: 'rotation',
                                  SphericalColorContrastGrating.motion_axis: 'vertical',
                                  SphericalColorContrastGrating.red01: c1,
                                  SphericalColorContrastGrating.green01: 0.0,
                                  SphericalColorContrastGrating.blue01: 0.0,
                                  SphericalColorContrastGrating.red02: 0.0,
                                  SphericalColorContrastGrating.green02: c2,
                                  SphericalColorContrastGrating.blue02: 0.0,
                                  })
                self.add_phase(phase)

                phase = vxprotocol.Phase(duration=4)
                phase.set_visual(SphericalColorContrastGrating,
                                 {SphericalColorContrastGrating.angular_period: angular_period_degrees,
                                  SphericalColorContrastGrating.angular_velocity: d * angular_velocity_degrees,
                                  SphericalColorContrastGrating.waveform: 'rectangular',
                                  SphericalColorContrastGrating.motion_type: 'rotation',
                                  SphericalColorContrastGrating.motion_axis: 'vertical',
                                  SphericalColorContrastGrating.red01: c1,
                                  SphericalColorContrastGrating.green01: 0.0,
                                  SphericalColorContrastGrating.blue01: 0.0,
                                  SphericalColorContrastGrating.red02: 0.0,
                                  SphericalColorContrastGrating.green02: c2,
                                  SphericalColorContrastGrating.blue02: 0.0,
                                  })
                self.add_phase(phase)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)
        