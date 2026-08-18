import os
import traceback

import torch
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

from wtforms import FileField, SubmitField, FloatField
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


app = Flask(__name__)

app.config["SECRET_KEY"] = "supersecretkey"
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

Bootstrap(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")
    alpha = FloatField("Alpha", default=1.0)
    submit = SubmitField("Transfer Style")


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 60)


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VGG_PATH = os.path.join(
    BASE_DIR,
    "vgg_normalised.pth"
)

DECODER_PATH = os.path.join(
    BASE_DIR,
    "experiment",
    "final_exp",
    "decoder_final.pth"
)

print("VGG MODEL:")
print(VGG_PATH)

print("DECODER MODEL:")
print(DECODER_PATH)

print("=" * 60)


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading VGG encoder...")

encoder = VGGEncoder(
    VGG_PATH
).to(device)

encoder.eval()

print("VGG encoder loaded successfully.")

print("Loading decoder...")

decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        DECODER_PATH,
        map_location=device
    )
)

decoder.eval()

print("Decoder loaded successfully.")

print("=" * 60)
print("ALL MODELS READY")
print("=" * 60)


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):

    if not filename:
        return False

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


# ============================================================
# STYLE TRANSFER
# ============================================================

def style_transfer(
    content_image,
    style_image,
    encoder,
    decoder,
    alpha,
    device
):

    print("Preparing images...")

    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    content_tensor = (
        content_transform(content_image)
        .unsqueeze(0)
        .to(device)
    )

    style_tensor = (
        style_transform(style_image)
        .unsqueeze(0)
        .to(device)
    )

    print(
        "Content tensor shape:",
        content_tensor.shape
    )

    print(
        "Style tensor shape:",
        style_tensor.shape
    )

    with torch.no_grad():

        print("Encoding content image...")

        content_feats = encoder(
            content_tensor,
            is_test=True
        )

        print("Encoding style image...")

        style_feats = encoder(
            style_tensor,
            is_test=True
        )

        print("Applying AdaIN...")

        stylized_feats = adaptive_instance_normalization(
            content_feats,
            style_feats
        )

        print("Applying style strength...")

        stylized_feats = (
            alpha * stylized_feats
            + (1 - alpha) * content_feats
        )

        print("Running decoder...")

        stylized_image = decoder(
            stylized_feats
        )

    print("Style transfer finished.")

    return stylized_image


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(image, path):

    print("Preparing output image...")

    image = image.detach().cpu().clone()

    image = image.squeeze(0)

    image = image.clamp(0, 1)

    image = transforms.ToPILImage()(image)

    # IMPORTANT:
    # Explicitly save as PNG so Pillow never has
    # to guess the file extension.

    image.save(
        path,
        format="PNG"
    )

    print("Output saved successfully:")
    print(path)


# ============================================================
# HOME / STYLE TRANSFER
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if request.method == "POST":

        print("\n")
        print("=" * 70)
        print("NEW STYLE TRANSFER REQUEST")
        print("=" * 70)

        try:

            # ------------------------------------------------
            # GET UPLOADED FILES
            # ------------------------------------------------

            content_file = request.files.get("content")
            style_file = request.files.get("style")

            print("Content file:", content_file)
            print("Style file:", style_file)

            # ------------------------------------------------
            # CHECK CONTENT IMAGE
            # ------------------------------------------------

            if not content_file:

                error = "Please upload a content image."

                print("ERROR:", error)

                return render_template(
                    "index.html",
                    form=form,
                    result_image=None,
                    content_image=None,
                    style_image=None,
                    error=error
                )

            if not content_file.filename:

                error = "Please select a content image."

                print("ERROR:", error)

                return render_template(
                    "index.html",
                    form=form,
                    result_image=None,
                    content_image=None,
                    style_image=None,
                    error=error
                )

            # ------------------------------------------------
            # CHECK STYLE IMAGE
            # ------------------------------------------------

            if not style_file:

                error = "Please upload a style image."

                print("ERROR:", error)

                return render_template(
                    "index.html",
                    form=form,
                    result_image=None,
                    content_image=None,
                    style_image=None,
                    error=error
                )

            if not style_file.filename:

                error = "Please select a style image."

                print("ERROR:", error)

                return render_template(
                    "index.html",
                    form=form,
                    result_image=None,
                    content_image=None,
                    style_image=None,
                    error=error
                )

            # ------------------------------------------------
            # CHECK FILE TYPES
            # ------------------------------------------------

            print(
                "Content filename:",
                content_file.filename
            )

            print(
                "Style filename:",
                style_file.filename
            )

            if not allowed_file(content_file.filename):

                error = (
                    "Invalid content image. "
                    "Please use PNG, JPG or JPEG."
                )

                print("ERROR:", error)

                return render_template(
                    "index.html",
                    form=form,
                    result_image=None,
                    content_image=None,
                    style_image=None,
                    error=error
                )

            if not allowed_file(style_file.filename):

                error = (
                    "Invalid style image. "
                    "Please use PNG, JPG or JPEG."
                )

                print("ERROR:", error)

                return render_template(
                    "index.html",
                    form=form,
                    result_image=None,
                    content_image=None,
                    style_image=None,
                    error=error
                )

            # ------------------------------------------------
            # SECURE FILENAMES
            # ------------------------------------------------

            content_original_name = secure_filename(
                content_file.filename
            )

            style_original_name = secure_filename(
                style_file.filename
            )

            # ------------------------------------------------
            # CREATE UNIQUE-ish FILE NAMES
            # ------------------------------------------------

            content_filename = (
                "content_" + content_original_name
            )

            style_filename = (
                "style_" + style_original_name
            )

            content_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                content_filename
            )

            style_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                style_filename
            )

            # ------------------------------------------------
            # SAVE UPLOADS
            # ------------------------------------------------

            print("Saving content image...")

            content_file.save(
                content_path
            )

            print(
                "Content saved:",
                content_path
            )

            print("Saving style image...")

            style_file.save(
                style_path
            )

            print(
                "Style saved:",
                style_path
            )

            # ------------------------------------------------
            # OPEN IMAGES
            # ------------------------------------------------

            print("Opening content image...")

            content_image = Image.open(
                content_path
            ).convert("RGB")

            print(
                "Content size:",
                content_image.size
            )

            print("Opening style image...")

            style_image = Image.open(
                style_path
            ).convert("RGB")

            print(
                "Style size:",
                style_image.size
            )

            # ------------------------------------------------
            # GET ALPHA
            # ------------------------------------------------

            alpha = form.alpha.data

            if alpha is None:
                alpha = 1.0

            alpha = float(alpha)

            # Keep alpha between 0 and 1

            alpha = max(
                0.0,
                min(
                    1.0,
                    alpha
                )
            )

            print(
                "Style strength:",
                alpha
            )

            # ------------------------------------------------
            # RUN STYLE TRANSFER
            # ------------------------------------------------

            print("Starting neural style transfer...")

            stylized_image = style_transfer(
                content_image,
                style_image,
                encoder,
                decoder,
                alpha,
                device
            )

            # ------------------------------------------------
            # CREATE PNG OUTPUT NAME
            # ------------------------------------------------

            # Remove the original extension completely.

            base_name = os.path.splitext(
                content_original_name
            )[0]

            result_filename = (
                "stylized_"
                + base_name
                + ".png"
            )

            result_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                result_filename
            )

            print(
                "Result filename:",
                result_filename
            )

            print(
                "Result path:",
                result_path
            )

            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            save_image(
                stylized_image,
                result_path
            )

            result_image = result_filename

            print("=" * 70)
            print("STYLE TRANSFER SUCCESSFUL")
            print(
                "RESULT:",
                result_image
            )
            print("=" * 70)

        except Exception as e:

            print("\n")
            print("=" * 70)
            print("NST ERROR")
            print("=" * 70)

            print(
                "Error type:",
                type(e).__name__
            )

            print(
                "Error:",
                str(e)
            )

            print("\nFULL TRACEBACK:")

            traceback.print_exc()

            print("=" * 70)

            error = (
                f"{type(e).__name__}: {str(e)}"
            )

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


# ============================================================
# SERVE UPLOADED IMAGES
# ============================================================

@app.route("/uploads/<filename>")
def send_image(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ============================================================
# SERVE EXAMPLE IMAGES
# ============================================================

@app.route("/examples/<path:filename>")
def send_example(filename):

    return send_from_directory(
        os.path.join(
            BASE_DIR,
            "examples"
        ),
        filename
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("NEURALART - ADAIN STYLE TRANSFER")
    print("=" * 70)

    print(
        "Open your browser at:"
    )

    print(
        "http://localhost:5000"
    )

    print(
        "Device:",
        device
    )

    print("=" * 70)

    app.run(
        host="localhost",
        port=5000,
        debug=True
    )