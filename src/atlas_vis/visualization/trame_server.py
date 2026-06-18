import logging
from typing import Any

import pyvista as pv
import requests
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import html, vuetify3
from trame.widgets import vtk as vtk_widgets

logger = logging.getLogger(__name__)


class TrameStreamingServer:
    def __init__(self, port: int = 8080, api_url: str = "http://127.0.0.1:8000", rendering_mode: str = "local", debug: bool = False) -> None:
        self.server = get_server()
        self.rendering_mode = rendering_mode
        self.debug = debug
        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug(f"Trame UI initialized in DEBUG mode. Rendering mode: {self.rendering_mode}")
        self.port = port
        self.api_url = api_url
        self.session = requests.Session()

        # PyVista plot state
        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.set_background("white")
        
        # Add a simple light source in the south
        sun = pv.Light(position=(0, -10000, 10000), focal_point=(0, 0, 0), light_type="scene light")
        sun.positional = False
        sun.intensity = 1.0
        self.plotter.add_light(sun)
        
        # Enable terrain style natively on the python side as well
        try:
            self.plotter.enable_terrain_style()
        except Exception:
            pass
        
        self._camera_initialized = False

        # Core UI state models
        self.server.state.update(
            {
                "trame__title": "AtlasVis",
                # Main Ribbon State
                "folders": [],
                "selected_folder": None,
                "files": [],
                "selected_file": None,
                "fields": [],
                "active_fields": [],
                # Interactive File Manager Dialogue State
                "folder_dialog": False,
                "browser_current_path": "",
                "browser_parent_path": "",
                "browser_dirs": [],
                "browser_files": [],
                "browser_marked_folder": None,
                "settings_dialog": False,
            }
        )

        self._setup_callbacks()
        self._setup_ui()

        # Seed the file manager state from the backend execution directory at launch
        self._fetch_directory_node()

    def _move_camera(self, direction: str) -> None:
        import numpy as np
        cam = self.plotter.camera
        
        pos = np.array(cam.position)
        foc = np.array(cam.focal_point)
        up = np.array(cam.up)
        
        view_vec = foc - pos
        distance = np.linalg.norm(view_vec)
        if distance == 0:
            return
            
        view_vec = view_vec / distance
        right_vec = np.cross(view_vec, up)
        right_vec_norm = np.linalg.norm(right_vec)
        if right_vec_norm != 0:
            right_vec = right_vec / right_vec_norm
            
        # Move step size relative to the zoom distance (5% per keypress)
        step = distance * 0.05
        
        if direction in ['w', 'W', 'ArrowUp']:
            move_vec = view_vec * step
        elif direction in ['s', 'S', 'ArrowDown']:
            move_vec = -view_vec * step
        elif direction in ['a', 'A', 'ArrowLeft']:
            move_vec = -right_vec * step
        elif direction in ['d', 'D', 'ArrowRight']:
            move_vec = right_vec * step
        else:
            return
            
        # Auto-level the horizon on every move
        cam.up = (0, 0, 1)
        cam.position = tuple(pos + move_vec)
        cam.focal_point = tuple(foc + move_vec)
        self.server.controller.view_update()
        if hasattr(self.server.controller, 'view_push_camera'):
            self.server.controller.view_push_camera()

    def _fix_roll(self) -> None:
        """Snaps the camera horizon to flat (Z-axis up) to fix vtk.js trackball rolling."""
        cam = self.plotter.camera
        cam.up = (0, 0, 1)
        self.server.controller.view_update()
        if hasattr(self.server.controller, 'view_push_camera'):
            self.server.controller.view_push_camera()

    def _fetch_directory_node(self, target_path: str | None = None) -> None:
        """Query the FastAPI backend to read the absolute filesystem node layout."""
        import time

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.session.get(f"{self.api_url}/api/browse", params={"target_path": target_path})
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        state = self.server.state
                        state.browser_current_path = data["current_path"]
                        state.browser_parent_path = data["parent_path"]
                        state.browser_dirs = data["directories"]
                        state.browser_files = data["files"]
                        state.browser_marked_folder = None
                        return
                else:
                    logger.error(f"Directory syncing error payload returned: {response.text}")
                    return
            except requests.exceptions.ConnectionError as exc:
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Wait for uvicorn to start
                else:
                    logger.error(
                        f"Failed connection sequence to backend API router after {max_retries} attempts: {exc}"
                    )
            except Exception as exc:
                logger.error(f"Unexpected error in _fetch_directory_node: {exc}")
                return

    def navigate_to_dir(self, dirname: str) -> None:
        """Advance down into selected child directory paths."""
        current = self.server.state.browser_current_path
        separator = "\\" if "\\" in current else "/"  # Handle clean cross-platform path joining
        next_path = f"{current.rstrip(separator)}{separator}{dirname}"
        self._fetch_directory_node(next_path)

    def navigate_to_parent(self) -> None:
        """Ascend up one directory level."""
        self._fetch_directory_node(self.server.state.browser_parent_path)

    def navigate_to_home(self) -> None:
        """Return to the user's home directory."""
        self._fetch_directory_node(None)

    def handle_folder_click(self, dirname: str) -> None:
        """Handles single and repeated clicks on a folder in the list."""
        state = self.server.state
        if state.browser_marked_folder == dirname:
            # The folder is already marked, so this second click chooses it
            self.navigate_to_dir(dirname)
        else:
            # First click: Just mark/highlight it
            state.browser_marked_folder = dirname

    def commit_selected_folder(self) -> None:
        """Appends the targeted directory to the operational ribbon list."""
        state = self.server.state
        current = state.browser_current_path
        marked = state.browser_marked_folder
        separator = "\\" if "\\" in current else "/"

        # Logic: If marked, use marked. If no folder is marked, use the parent view.
        if marked:
            chosen = f"{current.rstrip(separator)}{separator}{marked}"
        else:
            chosen = current

        if chosen and chosen not in state.folders:
            state.folders = [*state.folders, chosen]

        state.folder_dialog = False

    def remove_workspace_folder(self, folder_path: str) -> None:
        """Removes a folder from the active workspace list and clears dependent state."""
        state = self.server.state
        state.folders = [f for f in state.folders if f != folder_path]

        # FIX: Explicitly force Trame to push the updated list to the web UI
        state.dirty("folders")

        if state.selected_folder == folder_path:
            state.selected_folder = None
            state.fields = []

    def _setup_callbacks(self) -> None:
        state = self.server.state

        @state.change("folder_dialog")
        def on_dialog_open(folder_dialog, **kwargs):
            if folder_dialog and not state.browser_current_path:
                self._fetch_directory_node()

        @state.change("folders")
        def load_all_files(folders, **kwargs):
            """Aggregates all files from all selected workspace folders into a single list."""
            if not folders:
                state.files = []
                return

            aggregated_files = []
            for fld in folders:
                try:
                    res = self.session.get(f"{self.api_url}/api/browse", params={"target_path": fld})
                    if res.status_code == 200 and res.json().get("status") == "success":
                        # Extract just the folder name for clean UI display
                        separator = "\\" if "\\" in fld else "/"
                        fld_name = fld.split(separator)[-1]

                        for f_obj in res.json()["files"]:
                            f_name = f_obj["name"]
                            f_type = f_obj["type"]
                            full_path = f"{fld.rstrip(separator)}{separator}{f_name}"
                            aggregated_files.append(
                                {
                                    "id": full_path,  # Unique identifier used for backend parsing
                                    "name": f_name,  # Display name
                                    "parent_dir": fld_name,  # Shows the user which folder it came from
                                    "type": f_type,
                                }
                            )
                except Exception as e:
                    logger.error(f"Failed asynchronous fetch for {fld}: {e}")

            state.files = aggregated_files

        @state.change("selected_file")
        def load_file_metadata(selected_file, **kwargs):
            if not selected_file:
                return

            errors = dict(state.file_errors) if state.file_errors else {}

            try:
                if self.debug:
                    logger.debug(f"Attempting to fetch metadata from {self.api_url}/api/metadata for {selected_file}")
                res = self.session.get(f"{self.api_url}/api/metadata", params={"filepath": selected_file}, timeout=5)
                data = res.json()
                if self.debug:
                    logger.debug(f"Metadata response: {res.status_code} - {data}")
                if res.status_code == 200 and data.get("status") == "success":
                    fields = data.get("variables", [])
                    
                    # Synthesize wind vector if components exist
                    has_speed = any(f.get("name") == "wind_speed" for f in fields)
                    has_dir = any(f.get("name") == "wind_direction" for f in fields)
                    if has_speed and has_dir:
                        # Remove the separate components and replace with unified 'wind'
                        fields = [f for f in fields if f.get("name") not in ("wind_speed", "wind_direction")]
                        fields.append({"name": "wind", "type": "vector"})
                        
                    state.fields = fields
                    if selected_file in errors:
                        del errors[selected_file]
                else:
                    state.fields = []
                    errors[selected_file] = data.get("message", "Unknown parsing error.")
            except Exception as e:
                state.fields = []
                errors[selected_file] = f"Connection failed: {str(e)}"

            state.file_errors = errors

    def add_active_field(self, field_name: str) -> None:
        current = self.server.state.active_fields
        if not any(f["name"] == field_name for f in current):
            new_entry = {
                "name": field_name,
                "visible": True,
                "stride_x": 4,
                "stride_y": 4,
                "stride_z": 4,
                "temporal_stride": 1,
            }
            self.server.state.active_fields = [*current, new_entry]
            self.update_render()

    def remove_active_field(self, field_name: str) -> None:
        self.server.state.active_fields = [f for f in self.server.state.active_fields if f["name"] != field_name]
        self.update_render()

    def flush_and_render(self, active_fields: list[dict[str, Any]]) -> None:
        self.server.state.active_fields = active_fields
        self.update_render()

    def update_render(self) -> None:
        """Triggered when active fields change or stride sliders move."""
        state = self.server.state
        active_fields = state.active_fields
        selected_file = state.selected_file

        self.plotter.clear_actors()

        if not selected_file:
            self.server.controller.view_update()
            return

        is_station = selected_file.endswith(".dat") or selected_file.endswith(".csv")

        if not active_fields and not is_station:
            self.server.controller.view_update()
            return

        try:
            import numpy as np
            import rioxarray
            from pyproj import Transformer
            from atlas_vis.core.config import ConfigurationManager
            from atlas_vis.parsers.tif import TifParser
            from atlas_vis.parsers.pandas import GeoSphere
            import metpy.calc as mpcalc
            from metpy.units import units

            config = ConfigurationManager()._state
            center_lon = config.domain_center_lon
            center_lat = config.domain_center_lat
            size_x = config.domain_size_x_km * 1000.0
            size_y = config.domain_size_y_km * 1000.0

            # Attempt to find base topography if the selected file is a station
            is_station = selected_file.endswith(".dat") or selected_file.endswith(".csv")
            topo_file = None
        
            if is_station:
                for f in state.files:
                    if f["id"].endswith(".tif"):
                        topo_file = f["id"]
                        break
            else:
                topo_file = selected_file

            topo_ds = None
            transformer = None
            cx, cy = 0.0, 0.0
            topo_grid = None

            if topo_file:
                parser = TifParser()
                topo_ds = parser.load_dataset(topo_file)
            
                if topo_ds is not None and hasattr(topo_ds, "rio") and topo_ds.rio.crs is not None:
                    transformer = Transformer.from_crs("EPSG:4326", topo_ds.rio.crs, always_xy=True)
                    cx, cy = transformer.transform(center_lon, center_lat)

                    half_x = size_x / 2.0
                    half_y = size_y / 2.0

                    topo_ds = topo_ds.rio.clip_box(
                        minx=cx - half_x,
                        miny=cy - half_y,
                        maxx=cx + half_x,
                        maxy=cy + half_y,
                    )
                
                    # Render the topography mesh if it's either the selected file, or a base for the station
                    # Only use the first active field for the topology mesh if we're rendering topography directly
                    if not is_station:
                        topo_fields = active_fields
                    else:
                        # If it's a station, we just need the raw elevation to render the base mesh
                        topo_fields = []
                        if "height" in topo_ds.data_vars:
                            topo_fields = [{"name": "height", "visible": True, "stride_x": 4, "stride_y": 4}]
                
                    for field in topo_fields:
                        if not field.get("visible", True):
                            continue
                        clean_name = field["name"].strip("[]")
                        if clean_name not in topo_ds.data_vars:
                            continue
                        
                        stride_x = int(field.get("stride_x", 4))
                        stride_y = int(field.get("stride_y", 4))

                        dims = topo_ds[clean_name].dims
                        if len(dims) >= 2:
                            y_dim, x_dim = dims[-2], dims[-1]
                            isel_dict = {x_dim: slice(None, None, stride_x), y_dim: slice(None, None, stride_y)}
                            for d in dims[:-2]:
                                isel_dict[d] = 0
                            sub_ds = topo_ds.isel(isel_dict)
                        else:
                            sub_ds = topo_ds

                        if len(dims) >= 2:
                            z = sub_ds[clean_name].values
                            z = np.nan_to_num(z, nan=0.0)
                            z = np.squeeze(z)

                            x = sub_ds[x_dim].values - cx
                            y = sub_ds[y_dim].values - cy

                            xx, yy = np.meshgrid(x, y)
                            if z.shape != xx.shape:
                                if z.size == xx.size:
                                    z = z.reshape(xx.shape)
                                else:
                                    raise ValueError(f"CRITICAL Shape mismatch: xx={xx.shape}, z={z.shape}")

                            import pyvista as pv
                            topo_grid = pv.StructuredGrid(xx, yy, z)
                            self.plotter.add_mesh(topo_grid, color="white", show_edges=True)
                            break

            if is_station:
                station_parser = GeoSphere()
                ds = station_parser.load_dataset(selected_file)
                if ds is not None:
                    # Apply time aggregation
                    ds = ds.mean(dim="time", skipna=True)
                
                    # Get local coordinates
                    s_lat = ds.attrs.get("Latitude", center_lat)
                    s_lon = ds.attrs.get("Longitude", center_lon)
                    s_elev = ds.attrs.get("Elevation_m", 0.0)
                
                    if transformer:
                        s_cx, s_cy = transformer.transform(s_lon, s_lat)
                        s_x = s_cx - cx
                        s_y = s_cy - cy
                    else:
                        s_x, s_y = 0.0, 0.0
                
                    # Find topography height
                    topo_h = 0.0
                    if topo_grid is not None:
                        # PyVista probe to find Z
                        import pyvista as pv
                        probe_point = pv.PolyData(np.array([[s_x, s_y, 0.0]]))
                        # Project ray downwards and upwards
                        start = np.array([s_x, s_y, 10000.0])
                        end = np.array([s_x, s_y, -10000.0])
                        pts, ind = topo_grid.extract_surface().ray_trace(start, end)
                        if len(pts) > 0:
                            topo_h = pts[0][2]
                        else:
                            # Fallback to 0.0 if ray trace fails (no topography directly under station)
                            topo_h = 0.0
                    else:
                        # If no topo, use 0.0 so absolute height evaluates to normal height of station
                        topo_h = 0.0
                
                    import xarray as xr
                    ds["topography_height"] = xr.DataArray(topo_h)
                    ds["Elevation_m"] = xr.DataArray(s_elev)
                
                    from atlas_vis.physics.registry import HierarchicalPhysicsRegistry
                    reg = HierarchicalPhysicsRegistry()
                
                    deps, func = reg.get_best_equation("absolute_height", "global")
                    abs_h = func(ds, *deps)
                
                    # Combine winds
                    if "wind_speed" in ds.data_vars and "wind_direction" in ds.data_vars:
                        u, v = mpcalc.wind_components(
                            ds["wind_speed"].values * units("m/s"),
                            ds["wind_direction"].values * units.deg
                        )
                        u_val = float(np.squeeze(u.magnitude))
                        v_val = float(np.squeeze(v.magnitude))
                        wind_vec = np.array([u_val, v_val, 0.0])
                        ds["wind"] = xr.DataArray(wind_vec, dims=["component"])

                    # Render active fields
                    station_loc = np.array([s_x, s_y, float(abs_h.values)])
                
                    import pyvista as pv
                    for field in active_fields:
                        if not field.get("visible", True):
                            continue
                    
                        name = field["name"].strip("[]")
                        if name == "wind" and "wind" in ds.data_vars:
                            wv = ds["wind"].values
                            direction = wv / (np.linalg.norm(wv) + 1e-9)
                            # Create arrow
                            arrow = pv.Arrow(start=station_loc, direction=direction, scale=np.linalg.norm(wv)*10)
                            self.plotter.add_mesh(arrow, color="red")
                        elif name in ds.data_vars:
                            val = float(np.squeeze(ds[name].values))
                            sphere = pv.Sphere(radius=50, center=station_loc)
                            self.plotter.add_mesh(sphere, color="blue") # We could map color to value later
                        
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Render pipeline failed: {e}", exc_info=True)

        if not self._camera_initialized:
            self.plotter.camera_position = "xz"
            self.plotter.camera.elevation = 45
            self.plotter.reset_camera()
            self._camera_initialized = True

        self.server.controller.view_update()
        if hasattr(self.server.controller, 'view_push_camera'):
            self.server.controller.view_push_camera()

    def _setup_ui(self) -> None:
        state, ctrl = self.server.state, self.server.controller

        with SinglePageLayout(self.server) as layout:
            layout.title.set_text("AtlasVis")
            layout.toolbar.density = "compact"
            layout.toolbar.height = 30
            layout.toolbar.color = "blue-darken-4"
            layout.toolbar.theme = "dark"
            with layout.toolbar:
                vuetify3.VSpacer()
                with vuetify3.VBtn(icon=True, click="settings_dialog = true", size="x-small"):
                    vuetify3.VIcon("mdi-cog", size="small")

            # INTERACTIVE DIALOG WINDOW FOR DIRECTORY SELECTION
            with vuetify3.VDialog(v_model=("folder_dialog", False), max_width="700px"):
                with vuetify3.VCard(theme="dark", classes="bg-blue-darken-4"):
                    vuetify3.VCardTitle("Add System Workspace Folder", classes="bg-blue-darken-3 pa-4")

                    with vuetify3.VCardText(classes="pa-4"):
                        # Dynamic Breadcrumb indicator displaying exact host location
                        vuetify3.VTextField(
                            v_model=("browser_current_path", ""),
                            readonly=True,
                            label="Host Node Directory Focus",
                            density="compact",
                            prepend_inner_icon="mdi-folder-account",
                        )

                        # Directory contents window list
                        with vuetify3.VList(
                            classes="border mt-2 overflow-y-auto bg-blue-darken-3",
                            style="height: 300px;",
                        ):
                            # Home node element
                            with vuetify3.VListItem(
                                click=self.navigate_to_home,
                                prepend_icon="mdi-home",
                                title="~ (Home Directory)",
                            ):
                                pass

                            # Parents node element step-up
                            with vuetify3.VListItem(
                                click=self.navigate_to_parent,
                                prepend_icon="mdi-arrow-up-bold-box",
                                title=".. (Parent Directory)",
                            ):
                                pass

                            # Valid subfolders list matching
                            with vuetify3.VListItem(
                                v_for="(dir, idx) in browser_dirs",
                                key="idx",
                                # Point click to our Python function which handles single/double logic
                                click=(self.handle_folder_click, "[dir]"),
                                # Vuetify automatically highlights the item when active is true
                                active=("browser_marked_folder === dir",),
                                prepend_icon="mdi-folder",
                            ):
                                vuetify3.VListItemTitle("{{ dir }}")

                        # Secondary tracking panel showing structural files inside directory before adding
                        with vuetify3.VRow(classes="mt-2 px-3 align-center text-caption text-grey-lighten-1"):
                            vuetify3.VIcon("mdi-file-info-outline", size="small", classes="mr-1")
                            vuetify3.VLabel("Contains {{ browser_files.length }} structured data fields array matches.")

                    with vuetify3.VCardActions(classes="pa-4 pt-0"):
                        vuetify3.VSpacer()
                        vuetify3.VBtn("Cancel", click="folder_dialog = false", variant="text")
                        vuetify3.VBtn(
                            "Select Current Folder",
                            click=self.commit_selected_folder,
                            color="success",
                            variant="elevated",
                            prepend_icon="mdi-folder-check",
                        )

            # PRIMARY INTERACTIVE USER VIEWPORT
            with layout.content:
                with vuetify3.VContainer(fluid=True, classes="pa-0 fill-height d-flex flex-column"):
                    # CONTROL RIBBON
                    with vuetify3.VSheet(
                        elevation=1,
                        theme="dark",
                        classes="w-100 pa-1 flex-shrink-0 bg-blue-darken-4 border-b",
                    ):
                        with vuetify3.VRow(dense=True, style="height: 140px;"):
                            # Column 1: Workspace Folders
                            with vuetify3.VCol(cols="3", classes="d-flex flex-column h-100 position-relative"):
                                vuetify3.VBtn(
                                    icon="mdi-folder-plus",
                                    size="x-small",
                                    color="primary",
                                    variant="flat",
                                    click="folder_dialog = true",
                                    classes="position-absolute",
                                    style="top: 8px; right: 8px; z-index: 10;"
                                )
                                with vuetify3.VList(classes="border rounded overflow-y-auto flex-grow-1 bg-blue-darken-3 position-relative pa-1", density="compact"):
                                    html.Div(
                                        "WORKSPACES",
                                        style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); opacity: 0.15; font-size: 2rem; font-weight: bold; pointer-events: none; white-space: nowrap; letter-spacing: 2px;"
                                    )
                                    with vuetify3.VListItem(
                                        v_for="(fld, i) in folders",
                                        key="fld",
                                        density="compact",
                                        classes="bg-white rounded-lg mb-1 elevation-1 text-black",
                                        style="min-height: 24px;"
                                    ):
                                        vuetify3.VListItemTitle("{{ fld.split(/[\\\\/]/).pop() }}")
                                        vuetify3.VTooltip(
                                            activator="parent",
                                            location="bottom",
                                            text="{{ fld }}",
                                        )

                                        with vuetify3.Template(v_slot_append=True):
                                            with vuetify3.VBtn(
                                                icon="mdi-close",
                                                size="x-small",
                                                variant="text",
                                                color="grey",
                                                click=(
                                                    self.remove_workspace_folder,
                                                    "[fld]",
                                                ),
                                            ):
                                                pass

                            # Column 2: Extracted Files (Aggregated)
                            with vuetify3.VCol(cols="3", classes="d-flex flex-column h-100"):
                                with vuetify3.VList(classes="border rounded overflow-y-auto flex-grow-1 bg-blue-darken-3 position-relative pa-1", density="compact"):
                                    html.Div(
                                        "TARGET DATASETS",
                                        style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); opacity: 0.15; font-size: 2rem; font-weight: bold; pointer-events: none; white-space: nowrap; letter-spacing: 2px;"
                                    )
                                    vuetify3.VListItem(
                                        v_for="(item, i) in files",
                                        key="i",
                                        title=("item.name",),
                                        click="selected_file = item.id",
                                        active=("selected_file === item.id",),
                                        classes=(
                                            "['bg-white', 'rounded-lg', 'mb-1', 'elevation-1', "
                                            "(file_errors && file_errors[item.id]) ? 'text-red-darken-2' : (item.type === 'station' ? 'text-cyan-darken-3' : (item.type === 'raster' ? 'text-green-darken-3' : 'text-grey-darken-3'))]",
                                        ),
                                        density="compact",
                                        style="min-height: 24px;"
                                    )
                            # Column 3: Fields Manifestation (Double Click Action)
                            with vuetify3.VCol(cols="2", classes="d-flex flex-column h-100"):
                                with vuetify3.VList(
                                    classes="border rounded overflow-y-auto flex-grow-1 bg-blue-darken-3 text-caption position-relative pa-1", density="compact"
                                ):
                                    html.Div(
                                        "AVAILABLE FIELDS",
                                        style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); opacity: 0.15; font-size: 1.5rem; font-weight: bold; pointer-events: none; text-align: center; line-height: 1.2; letter-spacing: 2px;"
                                    )
                                    with html.Div(
                                        v_if="file_errors && file_errors[selected_file]",
                                        classes="text-red-lighten-1 pa-2 font-weight-bold bg-white rounded-lg ma-1",
                                    ):
                                        html.Div("Cannot parse this file:")
                                        html.Div("{{ file_errors[selected_file] }}", classes="font-weight-regular mt-1")

                                    vuetify3.VListItem(
                                        v_if="!(file_errors && file_errors[selected_file])",
                                        v_for="(fld, i) in fields",
                                        key="i",
                                        title=("fld.name",),
                                        click=(
                                            self.add_active_field,
                                            "['[' + fld.name + ']']",
                                        ),
                                        classes=("['bg-white', 'rounded-lg', 'mb-1', 'elevation-1', fld.mapped ? 'text-black' : 'text-red-darken-2']",),
                                        density="compact",
                                        prepend_icon="mdi-variable",
                                        style="min-height: 24px;"
                                    )

                            # Column 4: Staged Parameter Blocks
                            with vuetify3.VCol(
                                cols="4",
                                classes="d-flex flex-column h-100 overflow-y-auto",
                            ):
                                with vuetify3.VRow(dense=True, classes="ma-0"):
                                    with vuetify3.VCol(
                                        cols="12",
                                        v_for="(af, idx) in active_fields",
                                        key="idx",
                                        classes="pa-0",
                                    ):
                                        with vuetify3.VCard(
                                            variant="flat",
                                            classes="pa-1 border-b bg-white text-black rounded-0",
                                        ):
                                            with vuetify3.VRow(align="center", dense=True, classes="ma-0"):
                                                vuetify3.VCheckboxBtn(
                                                    v_model="af.visible",
                                                    change=(self.flush_and_render, "[active_fields]"),
                                                    density="compact",
                                                    color="success",
                                                    hide_details=True,
                                                )
                                                vuetify3.VCardTitle(
                                                    "{{ af.name }}",
                                                    classes="pa-0 text-body-2 font-weight-bold flex-grow-1",
                                                )

                                                with vuetify3.VMenu(
                                                    location="bottom end", close_on_content_click=False
                                                ):
                                                    with vuetify3.Template(v_slot_activator="{ props }"):
                                                        vuetify3.VBtn(
                                                            icon="mdi-cog",
                                                            density="compact",
                                                            variant="text",
                                                            color="grey",
                                                            v_bind="props",
                                                        )
                                                    with vuetify3.VCard(classes="pa-3", min_width="250px"):
                                                        html.Div(
                                                            "{{ selected_file ? selected_file.split(/[\\\\/]/).slice(-2, -1)[0] : '' }}",
                                                            classes="text-caption text-grey font-weight-bold mb-2 text-uppercase border-b pb-1"
                                                        )
                                                        vuetify3.VCardSubtitle("Stride Distance")
                                                        vuetify3.VSlider(
                                                            v_model="af.stride_x",
                                                            end=(self.flush_and_render, "[active_fields]"),
                                                            min=1,
                                                            max=8,
                                                            step=1,
                                                            label="X",
                                                            density="compact",
                                                            hide_details=True,
                                                            thumb_label=True,
                                                        )
                                                        vuetify3.VSlider(
                                                            v_model="af.stride_y",
                                                            end=(self.flush_and_render, "[active_fields]"),
                                                            min=1,
                                                            max=8,
                                                            step=1,
                                                            label="Y",
                                                            density="compact",
                                                            hide_details=True,
                                                            thumb_label=True,
                                                        )
                                                        vuetify3.VSlider(
                                                            v_model="af.stride_z",
                                                            end=(self.flush_and_render, "[active_fields]"),
                                                            min=1,
                                                            max=8,
                                                            step=1,
                                                            label="Z",
                                                            density="compact",
                                                            hide_details=True,
                                                            thumb_label=True,
                                                        )

                                                with vuetify3.VBtn(
                                                    icon="mdi-close",
                                                    density="compact",
                                                    variant="text",
                                                    color="grey",
                                                    click=(
                                                        self.remove_active_field,
                                                        "[af.name]",
                                                    ),
                                                ):
                                                    pass

                    # 3D RENDER CANVAS AREA
                    with vuetify3.VSheet(
                        classes="flex-grow-1 w-100 position-relative bg-white",
                        tabindex="0",
                        keydown=(self._move_camera, "[$event.key]"),
                        mouseup=self._fix_roll
                    ):
                        html_view = vtk_widgets.VtkRemoteLocalView(
                            self.plotter.render_window,
                            namespace="view",
                            mode=self.rendering_mode,
                            style="width: 100%; height: 100%; position: absolute;",
                            interactor_events=("InteractionEvent",),
                            InteractionEvent=self._fix_roll,
                        )
                        ctrl.view_update = html_view.update
                        ctrl.view_push_camera = html_view.push_camera

    def start(self) -> None:
        logger.info(f"Launching Unified Engine Interface on port {self.port}")
        self.server.start(port=self.port, host="0.0.0.0")
