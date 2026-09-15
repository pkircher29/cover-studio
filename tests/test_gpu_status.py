import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import httpx
# Status tests do not load/download transcription models. Real transcription
# and GPU generation are covered by the documented hardware/browser checks.
with patch.dict(sys.modules, {'sheetsage2_transcribe': types.ModuleType('sheetsage2_transcribe')}):
    import server


class GPUStatusTests(unittest.IsolatedAsyncioTestCase):
    async def state(self, loaded=False, backend='vulkan', models=None):
        def respond(request):
            if request.url.path == '/health':
                return httpx.Response(200, json={'status': 'ok', 'backend': backend, 'models': 2})
            return httpx.Response(200, json={'data': models if models is not None else [
                {'id': 'yue2', 'loaded': loaded}, {'id': 'asr', 'loaded': False}]})
        client = httpx.AsyncClient
        with patch.object(server.httpx, 'AsyncClient', side_effect=lambda **kw: client(
                transport=httpx.MockTransport(respond), **kw)):
            return await server.health()

    async def test_loaded_is_read_from_models_endpoint(self):
        with patch.dict(os.environ, ENGINE_BACKEND='vulkan', ENGINE_DEVICE_NAME='Intel Arc B580'):
            result = await self.state(loaded=True)
        self.assertEqual(result, {'ok': True, 'yue2_loaded': True,
                                 'backend': 'vulkan', 'device': 'Intel Arc B580'})

    async def test_unloaded_model_is_ready_for_lazy_loading(self):
        result = await self.state()
        self.assertTrue(result['ok'])
        self.assertFalse(result['yue2_loaded'])

    async def test_missing_model_is_not_ready(self):
        self.assertFalse((await self.state(models=[]))['ok'])

    async def test_backend_mismatch_does_not_claim_configured_gpu(self):
        with patch.dict(os.environ, ENGINE_BACKEND='vulkan', ENGINE_DEVICE_NAME='Intel Arc B580'):
            self.assertIsNone((await self.state(backend='cpu'))['device'])

    async def test_vulkan_does_not_show_unrelated_nvidia_memory(self):
        with patch.object(server, 'health', return_value={'ok': True, 'backend': 'vulkan'}), \
                patch.object(server, '_find_nvidia_smi') as find_smi:
            self.assertEqual(await server.gpu_vram(), {'available': False})
            find_smi.assert_not_called()


if __name__ == '__main__':
    unittest.main()
