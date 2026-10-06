"""CPU regression checks for the canonical PPP interface."""
import unittest
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mep_ldm.models.ppp import (
    PhysPropSurrogate, SurrogateConfig, SmallMLP, load_ppp, save_ppp,
)


class PPPInterfaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.model = PhysPropSurrogate(SurrogateConfig(box_bins=4, alpha_any=3.0))
        self.model.sv_calib = SmallMLP(1)
        self.model.d_calib = SmallMLP(1)
        self.x = torch.rand(2, 1, 8, 8, 8) * 0.8 + 0.1
        self.tmp = Path.cwd() / ('.ppp-test-' + uuid4().hex)
        self.tmp.mkdir()
        self.path = self.tmp / 'ppp.pt'
        self.addCleanup(self.cleanup_files)

    def cleanup_files(self):
        self.path.unlink(missing_ok=True)
        self.tmp.rmdir()

    def test_save_load_preserves_predictions_config_and_input_gradients(self):
        save_ppp(self.model, self.path)
        restored = load_ppp(self.path)
        self.assertEqual(restored.cfg, self.model.cfg)
        self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
        first = self.x.clone().requires_grad_()
        second = self.x.clone().requires_grad_()
        before = self.model(first, input_type='solid_probability')
        after = restored(second, input_type='solid_probability')
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        sum(v.sum() for v in before.values()).backward()
        sum(v.sum() for v in after.values()).backward()
        torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)

    def test_phase_convention_and_no_double_sigmoid(self):
        x = torch.ones(2, 1, 8, 8, 8)
        x[0] = 0
        expected = torch.tensor([0.9999, 0.0001])
        torch.testing.assert_close(self.model(x, input_type='solid_probability')['phi'], expected)
        probability = self.model(self.x, input_type='solid_probability')
        logits = self.model(torch.logit(self.x), input_type='logits')
        pore = self.model(1 - self.x, input_type='pore_probability')
        for key in probability:
            torch.testing.assert_close(probability[key], logits[key])
            torch.testing.assert_close(probability[key], pore[key])

    def test_frozen_predictor_and_decoder_allow_denoiser_gradients(self):
        save_ppp(self.model, self.path)
        predictor = load_ppp(self.path)
        denoiser = torch.nn.Conv3d(1, 1, 1)
        decoder = torch.nn.Conv3d(1, 1, 1)
        with torch.no_grad():
            denoiser.weight.fill_(0.2)
            denoiser.bias.fill_(0.1)
            decoder.weight.fill_(0.3)
            decoder.bias.fill_(0.2)
        decoder.requires_grad_(False)
        decoded = decoder(denoiser(self.x))
        predictions = predictor(decoded, input_type='solid_probability')
        loss = (predictions['phi'] - 0.2).square().mean()
        loss.backward()
        self.assertGreater(denoiser.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(denoiser.weight.grad).all())
        self.assertTrue(all(p.grad is None for p in decoder.parameters()))
        self.assertTrue(all(p.grad is None for p in predictor.parameters()))

    def test_existing_single_input_weights_require_explicit_settings(self):
        torch.save({'sv_calib': self.model.sv_calib.state_dict(),
                    'd_calib': self.model.d_calib.state_dict(), 'args': {}}, self.path)
        with self.assertRaisesRegex(ValueError, 'legacy_cfg'):
            load_ppp(self.path)
        migrated = load_ppp(self.path, legacy_cfg=self.model.cfg)
        for key, value in self.model(self.x).items():
            torch.testing.assert_close(value, migrated(self.x)[key], rtol=0, atol=0)
        save_ppp(migrated, self.path)
        load_ppp(self.path)

    def test_old_multi_input_surrogate_is_rejected(self):
        torch.save({'cfg': {}, 'state_dict': {'mlp_calib.net.0.weight': torch.ones(32, 2)}}, self.path)
        with self.assertRaisesRegex(ValueError, 'Incompatible legacy'):
            load_ppp(self.path)

    def test_corrupt_or_incomplete_weights_are_not_silently_loaded(self):
        save_ppp(self.model, self.path)
        checkpoint = torch.load(self.path, weights_only=True)
        checkpoint['state_dict'].pop('d_calib.net.0.weight')
        torch.save(checkpoint, self.path)
        with self.assertRaises(RuntimeError):
            load_ppp(self.path)

    def test_uncalibrated_model_cannot_be_exported_as_calibrated(self):
        with self.assertRaisesRegex(ValueError, 'Attach a trained'):
            save_ppp(PhysPropSurrogate(), self.path)

    def test_calibration_cache_and_export_use_same_operator(self):
        from mep_ldm.models.physical_property_predictor import collect_eval_cache, predict_from_cache
        labels = {'phi': torch.ones(2), 'S_v': torch.ones(2), 'D': torch.ones(2)}
        cache = collect_eval_cache([(self.x, labels)], self.model.cfg)
        from_cache = predict_from_cache(cache, self.model.sv_calib, self.model.d_calib)
        save_ppp(self.model, self.path)
        loaded = load_ppp(self.path)
        direct = loaded(self.x, input_type='solid_probability')
        for key in direct:
            torch.testing.assert_close(direct[key].cpu(), from_cache[key].cpu())

    def test_short_calibration_training_can_be_saved_and_loaded(self):
        from types import SimpleNamespace
        from mep_ldm.models.physical_property_predictor import collect_eval_cache, run_one_calibration
        labels = {'phi': (1 - self.x).mean((1, 2, 3, 4)),
                  'S_v': torch.tensor([0.3, 0.4]), 'D': torch.tensor([2.1, 2.3])}
        cache = collect_eval_cache([(self.x, labels)], self.model.cfg)
        args = SimpleNamespace(sv_mlp_epochs=2, sv_mlp_patience=2,
                               d_mlp_epochs=2, d_mlp_patience=2)
        sv_calib, d_calib, expected = run_one_calibration(cache, cache, args)
        calibrated = PhysPropSurrogate(self.model.cfg)
        calibrated.sv_calib = sv_calib
        calibrated.d_calib = d_calib
        save_ppp(calibrated, self.path)
        actual = load_ppp(self.path)(self.x, input_type='solid_probability')
        for key in actual:
            torch.testing.assert_close(actual[key].cpu(), expected[key].cpu())


if __name__ == '__main__':
    unittest.main()
