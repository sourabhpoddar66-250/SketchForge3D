"""CPU checks for the actual pipeline helpers; no generated examples are fabricated."""
import json
from pathlib import Path
import tempfile
import types
import unittest
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import trimesh
import engine


class PipelineChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def sketch(self, inverted=False):
        image = Image.new("RGB", (400, 240), "black" if inverted else "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 40, 300, 180), outline="white" if inverted else "black", width=4)
        path = self.root / ("inverted.png" if inverted else "sketch.png")
        image.save(path)
        return path

    def test_blank_is_rejected(self):
        source = self.root / "blank.png"
        Image.new("RGB", (64, 64), "white").save(source)
        with self.assertRaisesRegex(ValueError, "blank"):
            engine.condition_sketch(source, self.root)

    def test_control_map_shape_and_structure(self):
        stats = engine.condition_sketch(self.sketch(), self.root)
        control = np.asarray(Image.open(self.root / "02_control.png"))
        self.assertEqual(control.shape, (512, 512, 3))
        self.assertGreater(stats["stroke_fraction"], 0)
        self.assertLess(stats["stroke_fraction"], 0.2)
        # Framing preserves the rectangular proportions rather than stretching.
        y, x = np.nonzero(control[..., 0])
        self.assertAlmostEqual((x.max() - x.min()) / (y.max() - y.min()), 220 / 140, delta=0.07)

    def test_inverse_sketch_has_same_structure(self):
        a, b = self.root / "a", self.root / "b"
        a.mkdir()
        b.mkdir()
        engine.condition_sketch(self.sketch(), a)
        result = engine.condition_sketch(self.sketch(True), b)
        self.assertTrue(result["inverted_input"])
        np.testing.assert_array_equal(np.asarray(Image.open(a / "02_control.png")), np.asarray(Image.open(b / "02_control.png")))

    def test_transparent_background_is_white(self):
        image = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
        image.putpixel((10, 10), (0, 0, 0, 255))
        source = self.root / "alpha.png"
        image.save(source)
        rgb = np.asarray(engine.read_rgb(source))
        np.testing.assert_array_equal(rgb[0, 0], [255, 255, 255])
        np.testing.assert_array_equal(rgb[10, 10], [0, 0, 0])

    def test_small_disconnected_components_survive(self):
        body = trimesh.creation.box()
        detail = trimesh.creation.icosphere(subdivisions=1, radius=0.03)
        detail.apply_translation([0.7, 0, 0])
        joined = trimesh.util.concatenate([body, detail])
        cleaned = engine.clean_mesh(joined)
        self.assertEqual(len(cleaned.split(only_watertight=False)), 2)
        self.assertEqual(len(cleaned.faces), len(joined.faces))

    def test_simple_real_3d_geometry_is_accepted(self):
        report = engine.geometry_report(trimesh.creation.box())
        self.assertTrue(report["passed"])
        self.assertEqual(report["faces"], 12)

    def test_optional_decimation_keeps_geometry_and_colors(self):
        mesh = trimesh.creation.icosphere(subdivisions=3)
        mesh.visual.vertex_colors = np.tile([70, 140, 210, 255], (len(mesh.vertices), 1))
        reduced = engine.clean_mesh(mesh, target_faces=300)
        self.assertLess(len(reduced.faces), len(mesh.faces))
        self.assertTrue(engine.geometry_report(reduced)["passed"])
        self.assertEqual(reduced.visual.kind, "vertex")
        self.assertEqual(len(reduced.visual.vertex_colors), len(reduced.vertices))

    def test_flat_output_is_rejected(self):
        plane = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
        self.assertFalse(engine.geometry_report(plane)["passed"])

    def test_vertex_colors_survive_glb_round_trip(self):
        mesh = trimesh.creation.box()
        mesh.visual.vertex_colors = np.tile([20, 100, 210, 255], (len(mesh.vertices), 1))
        path = self.root / "colored.glb"
        mesh.export(path)
        check = engine.inspect_glb(path, expect_vertex_colors=True)
        self.assertTrue(check["passed"])
        self.assertTrue(check["has_vertex_colors"])
        self.assertFalse(engine.inspect_glb(path, expect_texture=True)["passed"])

    def test_uv_rasterization_matches_texture_coordinates(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        positions, mask = engine.rasterize_uv_positions(vertices, np.array([[0, 1, 2]]), vertices[:, :2], 32)
        ys, xs = np.nonzero(mask)
        np.testing.assert_allclose(positions[mask, 0], (xs + 0.5) / 32, atol=1e-6)
        np.testing.assert_allclose(positions[mask, 1], 1 - (ys + 0.5) / 32, atol=1e-6)

    def test_texture_bake_and_export_without_opengl(self):
        mesh = trimesh.creation.box()
        textured, texture = engine.bake_texture_cpu_atlas(mesh, 64, lambda points: points + 0.5)
        self.assertEqual(texture.size, (64, 64))
        self.assertEqual(textured.visual.kind, "texture")
        np.testing.assert_allclose(textured.bounds, mesh.bounds)
        path = self.root / "textured.glb"
        textured.export(path)
        check = engine.inspect_glb(path, expect_texture=True)
        self.assertTrue(check["passed"])
        self.assertTrue(check["has_texture"])
        engine.write_obj(textured, self.root)
        self.assertTrue((self.root / "model.obj").is_file())
        self.assertTrue((self.root / "model.mtl").is_file())
        self.assertGreater(len(list(self.root.glob("*.png"))), 0)

    def test_marching_cubes_adapter_preserves_asymmetric_axes(self):
        import mcubes
        xx, yy, zz = np.meshgrid(np.arange(24), np.arange(28), np.arange(32), indexing="ij")
        volume = ((xx - 7) / 3) ** 2 + ((yy - 12) / 5) ** 2 + ((zz - 18) / 7) ** 2
        expected, _ = mcubes.marching_cubes(volume, 1.0)
        adapted, faces = engine.marching_cubes_numpy(volume, 1.0)
        # The upstream helper reverses the adapter output's axis order.
        recovered = adapted[:, [2, 1, 0]]
        np.testing.assert_allclose(recovered.min(axis=0), expected.min(axis=0), atol=1e-5)
        np.testing.assert_allclose(recovered.max(axis=0), expected.max(axis=0), atol=1e-5)
        self.assertGreater(len(faces), 0)

    def test_failed_run_has_failed_report_and_diagnostic_bundle(self):
        source = self.root / "blank.png"
        Image.new("RGB", (64, 64), "white").save(source)
        output = self.root / "failed_run"
        args = types.SimpleNamespace(run_dir=str(output), quality="preview", prompt="", seed=42, candidates=1, control_strength=1.1, texture=False, require_texture=False, target_faces=0, obj=False, triposr_dir=str(self.root), input=str(source), product_image=False)
        self.assertEqual(engine.run_pipeline(args), 1)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertNotIn("glb", report)
        with zipfile.ZipFile(output / "result_bundle.zip") as archive:
            self.assertIn("report.json", archive.namelist())
            self.assertNotIn("model.glb", archive.namelist())


if __name__ == "__main__":
    unittest.main(verbosity=2)
