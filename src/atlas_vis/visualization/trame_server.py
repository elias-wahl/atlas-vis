from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3, vtk as vtk_widgets
import pyvista as pv
import logging

logger = logging.getLogger(__name__)

class TrameStreamingServer:
    """
    Orchestrates the VTK graphics pipeline through ASGI WebSockets to thin web clients.
    Leverages Vuetify 3 for a modern, responsive UI overlay integrated directly with the PyVista plotter.
    """
    def __init__(self, port: int = 8080) -> None:
        self.server = get_server()
        self.port = port
        self.plotter = pv.Plotter(off_screen=True)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """
        Assembles a Vuetify web window utilizing VtkRemoteLocalView.
        This provides adaptive streaming: sending raw geometry for high-bandwidth local connections,
        and falling back to compressed JPEG frame streaming for remote or constrained networks.
        """
        state, ctrl = self.server.state, self.server.controller
        
        with SinglePageLayout(self.server) as layout:
            with layout.toolbar:
                vuetify3.VCardTitle("atlas_vis - HPC Interactive Viewport")
                vuetify3.VSpacer()
                
                vuetify3.VCheckbox(
                    v_model=("viewMode", "local"),
                    on_icon="mdi-monitor-dashboard",
                    off_icon="mdi-server-network",
                    true_value="local",
                    false_value="remote",
                    hide_details=True,
                    dense=True,
                    label="Toggle Render Engine"
                )
                
            with layout.content:
                vuetify3.VContainer(fluid=True, classes="pa-0 fill-height"):
                    html_view = vtk_widgets.VtkRemoteLocalView(
                        self.plotter.render_window,
                        namespace="view",
                        mode=("viewMode", "local"), 
                        interactive_ratio=1
                    )
                    ctrl.view_update = html_view.update
                    ctrl.view_reset_camera = html_view.reset_camera

    def start(self) -> None:
        """Ignites the Trame underlying Tornado/Aiohttp web server on all host interfaces."""
        logger.info(f"Starting Trame Render Engine on port {self.port}")
        self.server.start(port=self.port, host="0.0.0.0")