import logging

import pyvista as pv
import requests
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify3

logger = logging.getLogger(__name__)


class TrameStreamingServer:
    def __init__(
        self, port: int = 8080, api_url: str = "http://127.0.0.1:8000"
    ) -> None:
        self.server = get_server()
        self.port = port
        self.api_url = api_url
        self.plotter = pv.Plotter(off_screen=True)

        # Core UI state models
        self.server.state.update(
            {
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

    def _fetch_directory_node(self, target_path: str | None = None) -> None:
        """Query the FastAPI backend to read the absolute filesystem node layout."""
        try:
            response = requests.get(
                f"{self.api_url}/api/browse", params={"target_path": target_path}
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    state = self.server.state
                    state.browser_current_path = data["current_path"]
                    state.browser_parent_path = data["parent_path"]
                    state.browser_dirs = data["directories"]
                    state.browser_files = data["files"]
                    state.browser_marked_folder = None
            else:
                logger.error(
                    f"Directory syncing error payload returned: {response.text}"
                )
        except Exception as exc:
            logger.error(f"Failed connection sequence to backend API router: {exc}")

    def navigate_to_dir(self, dirname: str) -> None:
        """Advance down into selected child directory paths."""
        current = self.server.state.browser_current_path
        separator = (
            "\\" if "\\" in current else "/"
        )  # Handle clean cross-platform path joining
        next_path = f"{current.rstrip(separator)}{separator}{dirname}"
        self._fetch_directory_node(next_path)

    def navigate_to_parent(self) -> None:
        """Ascend up one directory level."""
        self._fetch_directory_node(self.server.state.browser_parent_path)

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

        @state.change("folders")
        def load_all_files(folders, **kwargs):
            """Aggregates all files from all selected workspace folders into a single list."""
            if not folders:
                state.files = []
                return

            aggregated_files = []
            for fld in folders:
                try:
                    res = requests.get(
                        f"{self.api_url}/api/browse", params={"target_path": fld}
                    )
                    if res.status_code == 200 and res.json().get("status") == "success":
                        # Extract just the folder name for clean UI display
                        separator = "\\" if "\\" in fld else "/"
                        fld_name = fld.split(separator)[-1]

                        for f in res.json()["files"]:
                            full_path = f"{fld.rstrip(separator)}{separator}{f}"
                            aggregated_files.append(
                                {
                                    "id": full_path,  # Unique identifier used for backend parsing
                                    "name": f,  # Display name
                                    "parent_dir": fld_name,  # Shows the user which folder it came from
                                    "type": "model_output",
                                }
                            )
                except Exception as e:
                    logger.error(f"Failed asynchronous fetch for {fld}: {e}")

            state.files = aggregated_files

        @state.change("selected_file")
        def load_file_metadata(selected_file, **kwargs):
            if not selected_file:
                return

            try:
                # selected_file is now the full absolute path from our aggregated list
                res = requests.get(
                    f"{self.api_url}/api/metadata", params={"filepath": selected_file}
                )
                if res.status_code == 200:
                    state.fields = res.json().get("variables", [])
            except Exception:
                # Fallback mockup
                state.fields = ["POT_TEMP", "WIND_SPEED", "U", "V", "QVAPOR", "QCLOUD"]

    def add_active_field(self, field_name: str) -> None:
        current = self.server.state.active_fields
        if not any(f["name"] == field_name for f in current):
            new_entry = {
                "name": field_name,
                "visible": True,
                "spatial_stride": 1,
                "temporal_stride": 1,
            }
            self.server.state.active_fields = [*current, new_entry]

    def remove_active_field(self, field_name: str) -> None:
        self.server.state.active_fields = [
            f for f in self.server.state.active_fields if f["name"] != field_name
        ]

    def _setup_ui(self) -> None:
        state, ctrl = self.server.state, self.server.controller

        with SinglePageLayout(self.server) as layout:
            with layout.toolbar:
                vuetify3.VToolbarTitle("atlas_vis - Data Pipeline Dashboard")
                vuetify3.VSpacer()
                with vuetify3.VBtn(icon=True, click="settings_dialog = true"):
                    vuetify3.VIcon("mdi-cog")

            # INTERACTIVE DIALOG WINDOW FOR DIRECTORY SELECTION
            with vuetify3.VDialog(v_model=("folder_dialog", False), max_width="700px"):
                with vuetify3.VCard():
                    vuetify3.VCardTitle(
                        "Add System Workspace Folder", classes="bg-grey-lighten-3 pa-4"
                    )

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
                            classes="border mt-2 overflow-y-auto",
                            style="height: 300px;",
                        ):
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
                                # Point click and dblclick to our Python functions
                                click=(self.handle_folder_click, "[dir]"),
                                dblclick=(self.navigate_to_dir, "[dir]"),
                                # Vuetify automatically highlights the item when active is true
                                active=("browser_marked_folder === dir",),
                                prepend_icon="mdi-folder",
                            ):
                                vuetify3.VListItemTitle("{{ dir }}")

                        # Secondary tracking panel showing structural files inside directory before adding
                        with vuetify3.VRow(
                            classes="mt-2 px-3 align-center text-caption text-grey-darken-1"
                        ):
                            vuetify3.VIcon(
                                "mdi-file-info-outline", size="small", classes="mr-1"
                            )
                            vuetify3.VLabel(
                                "Contains {{ browser_files.length }} structured data fields array matches."
                            )

                    with vuetify3.VCardActions(classes="pa-4 pt-0"):
                        vuetify3.VSpacer()
                        vuetify3.VBtn(
                            "Cancel", click="folder_dialog = false", variant="text"
                        )
                        vuetify3.VBtn(
                            "Select Current Folder",
                            click=self.commit_selected_folder,
                            color="success",
                            variant="elevated",
                            prepend_icon="mdi-folder-check",
                        )

            # PRIMARY INTERACTIVE USER VIEWPORT
            with layout.content:
                with vuetify3.VContainer(
                    fluid=True, classes="pa-0 fill-height d-flex flex-column"
                ):
                    # CONTROL RIBBON
                    with vuetify3.VSheet(
                        elevation=1,
                        classes="w-100 pa-3 flex-shrink-0 bg-grey-lighten-5 border-b",
                    ):
                        with vuetify3.VRow(dense=True, style="height: 240px;"):
                            # Column 1: Workspace Folders
                            with vuetify3.VCol(
                                cols="3", classes="d-flex flex-column h-100"
                            ):
                                with vuetify3.VRow(
                                    dense=True, align="center", classes="pb-1"
                                ):
                                    vuetify3.VCardSubtitle(
                                        "Workspaces",
                                        classes="pa-0 font-weight-bold text-uppercase",
                                    )
                                    vuetify3.VSpacer()
                                    vuetify3.VBtn(
                                        icon="mdi-folder-plus",
                                        size="x-small",
                                        color="primary",
                                        variant="flat",
                                        click="folder_dialog = true",
                                    )

                                with vuetify3.VList(
                                    classes="border rounded overflow-y-auto flex-grow-1 bg-white"
                                ):
                                    with vuetify3.VListItem(
                                        v_for="(fld, i) in folders",
                                        key="fld",
                                        density="compact",
                                    ):
                                        vuetify3.VListItemTitle(
                                            "{{ fld.split(/[\\\\/]/).pop() }}"
                                        )
                                        vuetify3.VTooltip(
                                            activator="parent",
                                            location="bottom",
                                            text="{{ fld }}",
                                        )

                                        # --- The Event Bubbling Fix ---
                                        with vuetify3.Template(v_slot_append=True):
                                            # 'click_stop' intercepts the click so it doesn't trigger the list item underneath
                                            with vuetify3.VBtn(
                                                icon="mdi-close",
                                                size="small",
                                                variant="text",
                                                color="grey",
                                                click_stop=(
                                                    self.remove_workspace_folder,
                                                    "[fld]",
                                                ),
                                            ):
                                                pass

                            # Column 2: Extracted Files (Aggregated)
                            with vuetify3.VCol(
                                cols="3", classes="d-flex flex-column h-100"
                            ):
                                vuetify3.VCardSubtitle(
                                    "Target Datasets",
                                    classes="pb-1 font-weight-bold text-uppercase",
                                )
                                with vuetify3.VList(
                                    classes="border rounded overflow-y-auto flex-grow-1 bg-white"
                                ):
                                    vuetify3.VListItem(
                                        v_for="(item, i) in files",
                                        key="i",
                                        title=("item.name",),
                                        subtitle=(
                                            "item.parent_dir",
                                        ),  # Displays the origin folder underneath the filename
                                        click="selected_file = item.id",
                                        active=("selected_file === item.id",),
                                        base_color="blue-darken-2",
                                        density="compact",
                                    )
                            # Column 3: Fields Manifestation (Double Click Action)
                            with vuetify3.VCol(
                                cols="2", classes="d-flex flex-column h-100"
                            ):
                                vuetify3.VCardSubtitle(
                                    "Available Fields",
                                    classes="pb-1 font-weight-bold text-uppercase",
                                )
                                with vuetify3.VList(
                                    classes="border rounded overflow-y-auto flex-grow-1 bg-white text-caption"
                                ):
                                    vuetify3.VListItem(
                                        v_for="(fld, i) in fields",
                                        key="i",
                                        title=("fld",),
                                        dblclick=(
                                            self.add_active_field,
                                            "['[' + fld + ']']",
                                        ),
                                        density="compact",
                                        prepend_icon="mdi-variable",
                                    )

                            # Column 4: Staged Parameter Blocks
                            with vuetify3.VCol(
                                cols="4",
                                classes="d-flex flex-column h-100 overflow-y-auto",
                            ):
                                vuetify3.VCardSubtitle(
                                    "Pipeline Visual Configurations",
                                    classes="pb-1 font-weight-bold text-uppercase",
                                )
                                with vuetify3.VRow(dense=True, classes="ma-0"):
                                    with vuetify3.VCol(
                                        cols="12",
                                        v_for="(af, idx) in active_fields",
                                        key="idx",
                                        classes="pa-1",
                                    ):
                                        with vuetify3.VCard(
                                            variant="flat",
                                            classes="pa-2 border bg-white",
                                        ):
                                            with vuetify3.VRow(
                                                align="center", dense=True
                                            ):
                                                vuetify3.VIcon(
                                                    "mdi-layers-outline",
                                                    size="small",
                                                    classes="mr-2",
                                                    color="indigo",
                                                )
                                                vuetify3.VCardTitle(
                                                    "{{ af.name }}",
                                                    classes="pa-0 text-body-2 font-weight-bold flex-grow-1",
                                                )
                                                with vuetify3.VBtn(
                                                    icon="mdi-close",
                                                    density="compact",
                                                    variant="text",
                                                    color="grey",
                                                    click=(
                                                        self.remove_active_field,
                                                        "['[' + af.name + ']']",
                                                    ),
                                                ):
                                                    pass

                                            # Integrated Tuning Controls inside individual variable matrices
                                            with vuetify3.VRow(
                                                dense=True, classes="pt-1 align-center"
                                            ):
                                                with vuetify3.VCol(cols="4"):
                                                    vuetify3.VSwitch(
                                                        v_model=("af.visible",),
                                                        label="Render",
                                                        density="compact",
                                                        color="success",
                                                        hide_details=True,
                                                    )
                                                with vuetify3.VCol(cols="8"):
                                                    vuetify3.VSlider(
                                                        v_model=("af.spatial_stride",),
                                                        min=1,
                                                        max=8,
                                                        step=1,
                                                        label="Stride",
                                                        density="compact",
                                                        hide_details=True,
                                                        thumb_label=True,
                                                    )

                    # 3D RENDER CANVAS AREA
                    with vuetify3.VSheet(
                        classes="flex-grow-1 w-100 position-relative bg-black"
                    ):
                        html_view = vtk_widgets.VtkRemoteLocalView(
                            self.plotter.render_window,
                            namespace="view",
                            mode="local",
                            style="width: 100%; height: 100%; position: absolute;",
                        )
                        ctrl.view_update = html_view.update

    def start(self) -> None:
        logger.info(f"Launching Unified Engine Interface on port {self.port}")
        self.server.start(port=self.port, host="0.0.0.0")
