#!/usr/bin/env python3
import os
import io
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.transforms as mtransforms
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from pyproj import CRS, Transformer
import streamlit as st
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap

# -------------------------------------------------
# Custom colormap (smooth rainbow-like)
# -------------------------------------------------
colors_geosoft = [
    (0, 0, 255),     # blue
    (0, 255, 255),   # cyan
    (0, 255, 0),     # green
    (255, 255, 0),   # yellow
    (255, 127, 0),   # orange
    (255, 0, 0),     # red
    (255, 0, 255)    # magenta
]
colors_geosoft = [(r/255, g/255, b/255) for r, g, b in colors_geosoft]
custom_cmap = LinearSegmentedColormap.from_list("rainbow_like", colors_geosoft, N=256)




colors_surfer = [
    (0, 0, 64),      # very dark blue
    (0, 0, 128),     # dark blue
    (0, 0, 255),     # pure blue
    (0, 170, 255),   # sky blue
    (0, 255, 255),   # cyan
    (0, 255, 145),   # greenish cyan
    (0, 255, 63),    # green
    (182, 255, 0),   # yellow-green
    (255, 255, 0),   # yellow
    (255, 191, 0),   # yellow-orange
    (255, 127, 0),   # orange
    (255, 0, 0),     # red
    (255, 0, 255)    # magenta
]

# Normalize to [0,1]
colors_surfer = [(r/255, g/255, b/255) for r, g, b in colors_surfer]

# Create smooth rainbow-like colormap
custom_cmap_ = LinearSegmentedColormap.from_list("rainbow_like", colors_surfer, N=256)








# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def write_surfer_dsaa_grid(path: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> None:
    ny, nx = Z.shape
    x_min, x_max = float(np.nanmin(X)), float(np.nanmax(X))
    y_min, y_max = float(np.nanmin(Y)), float(np.nanmax(Y))
    z_min, z_max = float(np.nanmin(Z)), float(np.nanmax(Z))

    with open(path, "w", encoding="utf-8") as f:
        f.write("DSAA\n")
        f.write(f"{nx} {ny}\n")
        f.write(f"{x_min:.4f} {x_max:.4f}\n")
        f.write(f"{y_min:.4f} {y_max:.4f}\n")
        f.write(f"{z_min:.4f} {z_max:.4f}\n")
        if Y[1, 0] < Y[0, 0]:
            Z_to_write = np.flipud(Z)
        else:
            Z_to_write = Z
        for j in range(ny):
            row = Z_to_write[j, :]
            f.write(" ".join(
                ["nan" if np.isnan(v) else f"{float(v):.6f}" for v in row]) + "\n")

def add_scalebar_bottom(fig, ax, length_km: float = 0.1):
    bar_len_m = length_km * 1000.0
    x0, x1 = ax.get_xlim()
    frac = bar_len_m / (x1 - x0)
    x_center = 0.5
    y_offset = -0.25
    bar_height = 0.015
    ax.add_patch(plt.Rectangle(
        (x_center - frac / 2, y_offset),
        frac, bar_height,
        transform=ax.transAxes,
        color="k", clip_on=False
    ))
    ax.text(
        x_center, y_offset - 0.05,
        f"{length_km*1000:.0f} m",
        ha="center", va="top",
        transform=ax.transAxes,
        fontsize=7
    )

def add_north_arrow_small(ax):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    x = x1 - 0.02 * (x1 - x0)
    y = y1 - 0.002 * (y1 - y0)
    ax.annotate('N', xy=(x, y), xytext=(x, y - 0.08 * (y1 - y0)),
                ha='center', va='top', fontsize=12,
                arrowprops=dict(arrowstyle='-|>', lw=3, color="k"))

def grid_points_with_extrapolation(xs, ys, vals,
                                   nx=200, ny=200,
                                   method="linear",
                                   margin_frac=0.05):
    xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
    ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
    dx = xmax - xmin; dy = ymax - ymin
    xmin -= margin_frac * dx; xmax += margin_frac * dx
    ymin -= margin_frac * dy; ymax += margin_frac * dy

    X_lin = np.linspace(xmin, xmax, nx)
    Y_lin = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(X_lin, Y_lin)

    Z = griddata(np.column_stack([xs, ys]), vals, (X, Y), method=method)
    if np.isnan(Z).any():
        Z_near = griddata(np.column_stack([xs, ys]), vals, (X, Y), method="nearest")
        Z = np.where(np.isnan(Z), Z_near, Z)
    return X, Y, Z

def make_map_utm(Xutm, Yutm, Zutm, transformer_inv, df,
                 title="", levels=20, contour_labels=True,
                 cmap="turbo", smooth_sigma=2.5):
    fig, ax = plt.subplots(figsize=(9, 7), dpi=400)
    
        
    Zplot = gaussian_filter(Zutm, sigma=smooth_sigma) if smooth_sigma and smooth_sigma > 0 else Zutm
    raster = ax.pcolormesh(Xutm, Yutm, Zplot, shading='auto', cmap=cmap)

    
    try:
        CS = ax.contour(Xutm, Yutm, Zplot, levels=levels, linewidths=0.4, colors='k', alpha=0.9)
        if contour_labels:
            ax.clabel(CS, inline=True, fontsize=7, fmt="%g")
    except Exception:
        pass

    cbar = fig.colorbar(raster, ax=ax, shrink=0.9, pad=0.02)
    
    
    preset_labels = [
    "Apparent Resistivity (Ω·m)",
    "Conductivity (S/m)",
    "Chargeability (mV/V)",
    "Magnetic Susceptibility (SI)",
    "Magnetic Anomaly (nT)",
    "Gravity Anomaly(mG)",
    "Thickness (m)",
    "pressure (mb)"
    ]
    
    label_choice = st.selectbox("Select Colorbar Label", ["Custom"] + preset_labels,help="Select/Customise ColorBar **Label** based on your data type")
    if label_choice == "Custom":
        cbar_label = st.text_input("custom label", "My Custom Label",help="Customise ColorBar **Label** based on your data type")
    else:
        cbar_label = label_choice
    cbar.set_label(cbar_label)

    def format_lon(x, pos):
        lon, lat = transformer_inv.transform(x, ax.get_ylim()[0])
        suffix = "E" if lon >= 0 else "W"
        return f"{abs(lon):.4f}°{suffix}"

    def format_lat(y, pos):
        lon, lat = transformer_inv.transform(ax.get_xlim()[0], y)
        suffix = "N" if lat >= 0 else "S"
        return f"{abs(lat):.4f}°{suffix}"

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_verticalalignment("center")
        label.set_horizontalalignment("right")
        offset = mtransforms.ScaledTranslation(0, -10/72, fig.dpi_scale_trans)
        label.set_transform(label.get_transform() + offset)

    add_scalebar_bottom(fig, ax, length_km=0.1)
    add_north_arrow_small(ax)

    if "Priority" in df.columns:
        priority_lats = pd.to_numeric(df["Latitude"], errors='coerce').to_numpy()
        priority_lons = pd.to_numeric(df["Longitude"], errors='coerce').to_numpy()
        priority_labels = df["Priority"].astype(str).to_numpy()

        mask_priority = ~(np.isnan(priority_lats) | np.isnan(priority_lons) |
                          (priority_labels == '') | (priority_labels == 'nan'))
        if mask_priority.any():
            priority_xs, priority_ys = transformer_fwd.transform(priority_lons[mask_priority],
                                                                 priority_lats[mask_priority])
            priority_texts = priority_labels[mask_priority]
            ax.scatter(priority_xs, priority_ys, color='black', s=30, zorder=5)
            for x, y, label in zip(priority_xs, priority_ys, priority_texts):
                ax.text(x, y, label, color='red', fontsize=10, ha='left', va='bottom', zorder=6)

    wrapped_title = "\n".join(textwrap.wrap(title, width=60))
    ax.set_title(wrapped_title, pad=10)

    ax.set_aspect('equal', adjustable='box')
    
    ax.grid(
    which='both',
    color='grey',     # grid line color
    linestyle='--',   # style: dashed, dotted, etc.
    linewidth=0.8,    # thin line
    alpha=0.2         # very transparent
    )
    
    plt.subplots_adjust(left=0.15, right=0.96, top=0.90, bottom=0.22)
    return fig

def make_3d_plot(Xutm, Yutm, Zutm, cmap, title=""):
    fig3d = go.Figure(data=[go.Surface(z=Zutm, x=Xutm, y=Yutm, colorscale=cmap)])
    fig3d.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Resistivity"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig3d

# -------------------------------------------------
# Streamlit App
# -------------------------------------------------
def run():
    st.markdown("<h1 style='text-align: center;'>🗺️ Geophysical Data Mapping<br><span style='font-size: 0.6em;'>(Grid View)</span></h1>", unsafe_allow_html=True)
    #st.title("🗺️ Geophysical Data Mapping (Grid View)")

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"],help="upload your data file, if need input template format: download below")
    
    # Path to template excel file
    template_path = "template_Rudra Geophysicist.xlsx"

     # Download button
    with open(template_path, "rb") as file:
        st.download_button(
            label="📥 Download Input Template",
            data=file,
            help ="Input template",
            file_name="template_Rudra Geophysicist.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    map_help = """
                Enter the title of your map.  
                This text will appear at the top of the generated grid or image.  
                (e.g., *Apparent Resistivity Map*, *Magnetic Anomaly Map*, etc.).
               """
    map_title = st.text_input("Map Title", "Apparent Resistivity Map",help=map_help)
    help_text = """
                This sets the grid resolution for interpolation.  
                - **Lower values** (e.g., 20×20) → fast processing, coarse map.  
                - **Higher values** (e.g., 200×200) → smoother result, slower calculation.  
                Choose based on the size of your dataset and performance needs.
                """
    grid_nx = st.number_input("Grid NX", min_value=10, max_value=500, value=60,help=help_text)
    grid_ny = st.number_input("Grid NY", min_value=10, max_value=500, value=60,help=help_text)
    
    interp_help = """
                    Select an interpolation method to build the grid model:
                    - **nearest**: Fastest, but rough and blocky interpolation.
                    - **linear**: Most commonly used — smooth but may miss small details.
                    - **cubic**: Smoothest result, but slower, especially for large datasets.
                  """
    
    
    contour_help = """
                    Defines how many contour lines or intervals appear on the contour map.  
                    - Fewer levels (e.g., 5–10): simple visualization, good for large areas.  
                    - More levels (e.g., 25+): detailed map showing subtle variations.  
                    Choose based on data range and resolution for best results.
                   """
    interp_method = st.selectbox("Interpolation Method", ["nearest", "linear", "cubic"],help=interp_help)
    contour_option = st.radio("Contour Levels", ["Auto", "Interval"],help= contour_help)
    if contour_option == "Auto":
        contour_levels = 15
    else:
        interval = st.number_input("Contour Interval", min_value=1, value=10,help="Enter required contour intervals, must be integer")
        contour_levels = np.arange(0, 1000, interval)

    # ✅ colormap dictionary
    cmap_dict = {
        "turbo": "turbo",
        "Surfer_color": custom_cmap_,
        "Geosoft_tbl": custom_cmap,
        "hsv": "hsv",
        "viridis": "viridis",
        "plasma": "plasma",
        "cividis": "cividis",
        "inferno": "inferno",
        
    }
    colormap_help = """
                    Colormap determines how values are mapped to colors on the plot.
                    Select the required color ramp given.
                    """
    cmap_choice_label = st.selectbox("Colormap", list(cmap_dict.keys()),help=colormap_help)
    cmap_choice = cmap_dict[cmap_choice_label]

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        lats = pd.to_numeric(df["Latitude"], errors='coerce').to_numpy()
        lons = pd.to_numeric(df["Longitude"], errors='coerce').to_numpy()
        vals = pd.to_numeric(df["Resistivity"], errors='coerce').to_numpy()

        mask = ~(np.isnan(lats) | np.isnan(lons) | np.isnan(vals))
        lats, lons, vals = lats[mask], lons[mask], vals[mask]

        mean_lat, mean_lon = float(np.mean(lats)), float(np.mean(lons))
        utm_zone = int((mean_lon + 180) // 6) + 1
        crs_utm = CRS.from_string(
            f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs" + (" +south" if mean_lat < 0 else "")
        )
        global transformer_fwd
        transformer_fwd = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
        transformer_inv = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)

        xs, ys = transformer_fwd.transform(lons, lats)
        Xutm, Yutm, Zutm = grid_points_with_extrapolation(xs, ys, vals,
                                                          nx=grid_nx, ny=grid_ny,
                                                          method=interp_method)

        # Save Surfer grid
        grd_path = "resistivity.grd"
        write_surfer_dsaa_grid(grd_path, Xutm, Yutm, Zutm)

        # 2D Map
        fig2d = make_map_utm(Xutm, Yutm, Zutm, transformer_inv, df,
                             title=map_title, levels=contour_levels,
                             cmap=cmap_choice)
        st.pyplot(fig2d)

        # ✅ Convert custom colormap for Plotly
        if isinstance(cmap_choice, LinearSegmentedColormap):
            cmap_plotly = [
                [i/255, f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"]
                for i, (r, g, b, _) in enumerate(cmap_choice(np.linspace(0, 1, 256)))
            ]
        else:
            cmap_plotly = cmap_choice

        # 3D Interactive Map
        fig3d = make_3d_plot(Xutm, Yutm, Zutm, cmap_plotly, title=map_title)
        st.plotly_chart(fig3d, use_container_width=True)

        # Downloads
        buf_png = io.BytesIO()
        fig2d.savefig(buf_png, format="png", bbox_inches="tight")
        st.download_button("Download PNG", data=buf_png.getvalue(),
                           file_name="resistivity_map.png", mime="image/png")

        buf_pdf = io.BytesIO()
        fig2d.savefig(buf_pdf, format="pdf", bbox_inches="tight")
        st.download_button("Download PDF", data=buf_pdf.getvalue(),
                           file_name="resistivity_map.pdf", mime="application/pdf")

        with open(grd_path, "rb") as f:
            st.download_button("Download Surfer GRD", data=f,
                               file_name="resistivity.grd", mime="application/octet-stream")

        html_buf = io.StringIO()
        fig3d.write_html(html_buf)
        st.download_button("Download 3D HTML", data=html_buf.getvalue(),
                           file_name="resistivity_3d.html", mime="text/html")


if __name__ == "__main__":
    run()
