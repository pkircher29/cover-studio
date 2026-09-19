"""Routing and failure checks; real-model checks are run separately in the UI."""
import copy
import asyncio
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import sheetsage2_transcribe
import server
import session_store


class MelodyMergeTests(unittest.TestCase):
    def test_vocals_replace_only_vocal_notes_and_keep_original_timestamps(self):
        mix = [{"time": 0.5, "global_subbeat": 4, "values": {
            "rhythm": {"meter": [4, 4], "eighth_position": 0}, "key": "C:maj",
            "chord": "C:maj", "melody": [
                {"track": 0, "pitch": 40, "end_time": 1.0},
                {"track": 1, "pitch": 72, "end_time": 1.5}]}}]
        vocal = [{"time": 0.7, "global_subbeat": 8, "values": {
            "key": "D:maj", "melody": [
                {"track": 0, "pitch": 60, "end_time": 2.1},
                {"track": 1, "pitch": 99, "end_time": 1.2}]}}]
        originals = copy.deepcopy((vocal, mix))
        merged = sheetsage2_transcribe.merge_events(vocal, mix)
        self.assertEqual((vocal, mix), originals)
        self.assertEqual(merged[0]["values"]["key"], "C:maj")
        self.assertEqual(merged[0]["values"]["melody"][0]["pitch"], 72)
        self.assertEqual(merged[1]["values"], {"melody": [{"track": 0, "pitch": 60, "end_time": 2.1}]})
        self.assertEqual(merged[1]["time"], 0.7)


class PreparationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        saver = patch.object(server.session_store, "save")
        saver.start()
        self.addCleanup(saver.stop)
    async def test_both_paths_are_kept_and_generation_waits_for_preparation(self):
        session = server.Session("test", "original.mp3", "temp", "original.mp3")
        server.SESSIONS[session.id] = session
        try:
            with self.assertRaises(server.HTTPException) as error:
                await server.generate_covers(session.id, lyrics="hello", styles='["keep_original"]')
            self.assertEqual(error.exception.status_code, 409)
            job = server.Job("job", "transcribe_lyrics")
            score = {"abc": "melody score", "full_abc": "full score"}
            with patch.object(server.vocal_separation, "separate", AsyncMock(return_value="vocals.wav")), \
                 patch.object(server.whisper_transcribe, "transcribe", AsyncMock(return_value={"text": "hello", "words": [], "model": "whisper-large-v3"})) as asr, \
                 patch.object(server.sheetsage2_transcribe, "transcribe_cover", Mock(return_value=score), create=True) as melody:
                await server._run_lyrics_job(job, session)
            asr.assert_awaited_once_with("vocals.wav", "temp")
            self.assertEqual(melody.call_args.args[:2], ("vocals.wav", "original.mp3"))
            self.assertEqual(session.source_path, "original.mp3")
            self.assertTrue(session.ready)
            self.assertEqual(job.status, "done")
        finally:
            server.SESSIONS.pop(session.id, None)

    async def test_separation_failure_never_transcribes_the_mix_as_a_fallback(self):
        session = server.Session("failed", "original.wav", "temp", "original.wav")
        server.SESSIONS[session.id] = session
        job = server.Job("job", "transcribe_lyrics")
        with patch.object(server.vocal_separation, "separate", AsyncMock(side_effect=RuntimeError("failed"))), \
             patch.object(server.whisper_transcribe, "transcribe", AsyncMock()) as asr:
            await server._run_lyrics_job(job, session)
        asr.assert_not_awaited()
        self.assertFalse(session.ready)
        self.assertEqual(job.status, "error")
        self.assertIn(session.id, server.SESSIONS)
        self.assertIn("failed", session.error)
        server.SESSIONS.pop(session.id)

    async def test_batch_uses_prepared_score_without_transcribing_again(self):
        batch = server.Batch(id="test", styles=[server.StyleRun(id="keep_original", label="Original")])
        with tempfile.TemporaryDirectory() as root:
            work = Path(root) / "work"
            work.mkdir()
            with patch.object(server, "COMPLETED_DIR", Path(root)), \
                 patch.object(server, "_call_yue2_generate", AsyncMock(return_value=b"wav")) as generate:
                await server._run_batch(batch, "original.wav", str(work), "song.wav",
                                        "hello", "full", 1, 4, prepared_abc="combined score")
            self.assertEqual(batch.status, "done")
            body = generate.call_args.args[0]
            self.assertEqual(body["request"]["options"]["abc"], "combined score")

    async def test_reopening_ready_song_skips_all_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            vocals = Path(folder) / "vocals.wav"
            vocals.write_bytes(b"audio")
            session = server.Session("cached", "original.wav", folder, "song.wav",
                                     vocals_path=str(vocals), lyrics="Reviewed words",
                                     melody={"abc": "saved", "full_abc": "saved full"}, ready=True)
            with patch.object(server.vocal_separation, "separate", AsyncMock()) as separation, \
                 patch.object(server.whisper_transcribe, "transcribe", AsyncMock()) as asr, \
                 patch.object(server.sheetsage2_transcribe, "transcribe_cover", Mock(), create=True) as melody:
                job = server.Job("cached-job", "transcribe_lyrics")
                await server._run_lyrics_job(job, session)
            separation.assert_not_awaited()
            asr.assert_not_awaited()
            melody.assert_not_called()
            self.assertEqual(job.result["lyrics"], "Reviewed words")

    async def test_generate_twice_retains_the_song_and_records_both_runs(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "song"
            folder.mkdir()
            source = folder / "source.wav"
            source.write_bytes(b"original")
            session = server.Session("repeat", str(source), str(folder), "song.wav",
                ready=True, lyrics="hello", melody={"abc": "melody", "full_abc": "harmony"})
            server.SESSIONS[session.id] = session
            tasks = []
            create_task = asyncio.create_task
            def capture(coro):
                task = create_task(coro)
                tasks.append(task)
                return task
            try:
                with patch.object(server, "COMPLETED_DIR", Path(root)), \
                     patch.object(server, "_call_yue2_generate", AsyncMock(return_value=b"wav")), \
                     patch.object(server.asyncio, "create_task", side_effect=capture):
                    for seed in (1, 2):
                        await server.generate_covers(session.id, "reviewed", '["keep_original"]',
                                                     "full", seed, 4, False, "any")
                        await tasks[-1]
                self.assertEqual(len(session.generations), 2)
                self.assertTrue(all(run["status"] == "done" for run in session.generations))
                self.assertEqual(source.read_bytes(), b"original")
                self.assertIn(session.id, server.SESSIONS)
                self.assertTrue(session.ready)
                self.assertIsNone(session.active_batch_id)
            finally:
                server.SESSIONS.pop(session.id, None)


class SessionPersistenceTests(unittest.TestCase):
    def test_restart_preserves_reviewed_lyrics_score_and_original(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "song-123"
            (folder / "stems").mkdir(parents=True)
            original = folder / "source.mp3"
            original.write_bytes(b"original")
            vocals = folder / "stems/vocals.wav"
            vocals.write_bytes(b"vocals")
            session = session_store.Session("123", str(original), str(folder), "Song.mp3",
                vocals_path=str(vocals), lyrics="My reviewed lyrics", ready=True,
                melody={"abc": "saved score", "full_abc": "full score"},
                generations=[{"batch_id": "old", "status": "done", "styles": []}])
            session_store.save(session)
            loaded = session_store.load_all(Path(root))["123"]
            self.assertEqual(loaded.lyrics, "My reviewed lyrics")
            self.assertEqual(loaded.melody, session.melody)
            self.assertTrue(loaded.ready)
            self.assertEqual(Path(loaded.source_path).read_bytes(), b"original")
            self.assertEqual(loaded.generations, session.generations)


if __name__ == "__main__":
    unittest.main()
