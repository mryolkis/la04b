#include <algorithm>
#include <chrono>
#include <filesystem>
#include <foobar2000/SDK/foobar2000.h>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <thread>
#include <vector>

DECLARE_COMPONENT_VERSION("La decoder host tests", "0.1.0", "Test-only component");
VALIDATE_COMPONENT_FILENAME("foo_la_test.dll");

namespace
{
using Samples = std::vector<audio_sample>;
abort_callback_dummy aborter;

class CancelOnSecondCheck : public abort_callback
{
    mutable unsigned checks = 0;
    bool is_aborting() const override
    {
        return ++checks >= 2;
    }
    abort_callback_event get_abort_event() const override
    {
        return aborter.get_abort_event();
    }
};

void require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}
input_decoder::ptr open_decoder(const std::filesystem::path &path)
{
    const pfc::stringcvt::string_utf8_from_wide utf8(path.c_str());
    input_decoder::ptr decoder;
    pfc::string8 canonical;
    filesystem::g_get_canonical_path(utf8.get_ptr(), canonical);
    input_entry::g_open_for_decoding(decoder, nullptr, canonical, aborter);
    decoder->initialize(0, 0, aborter);
    return decoder;
}
Samples read_all(input_decoder::ptr decoder)
{
    audio_chunk_impl chunk;
    Samples pcm;
    while (decoder->run(chunk, aborter))
    {
        pcm.insert(pcm.end(), chunk.get_data(), chunk.get_data() + chunk.get_data_length());
    }
    return pcm;
}
void check_file(const std::filesystem::path &la, const std::filesystem::path &wav,
                std::ofstream &log)
{
    auto reference_decoder = open_decoder(wav);
    file_info_impl expected_info;
    reference_decoder->get_info(0, expected_info, aborter);
    auto expected = read_all(reference_decoder);
    auto decoder = open_decoder(la);
    file_info_impl info;
    decoder->get_info(0, info, aborter);
    require(std::string(info.info_get("codec")) == "La 0.4b", "wrong decoder selected");
    require(info.get_length() == expected_info.get_length(), "duration differs from WAV");
    require(info.info_get_int("channels") == expected_info.info_get_int("channels"),
            "channels differ");
    require(info.info_get_int("samplerate") == expected_info.info_get_int("samplerate"),
            "sample rate differs");
    require(decoder->can_seek(), "decoder reports no seek support");
    require(read_all(decoder) == expected, "full PCM differs from foobar2000 WAV decoder");
    const auto channels = static_cast<size_t>(info.info_get_int("channels"));
    const auto rate = static_cast<unsigned>(info.info_get_int("samplerate"));
    const auto frames = expected.size() / channels;
    audio_chunk_impl chunk;

    for (size_t frame : {frames / 2, size_t(1), frames, frames + 100, size_t(0), frames - 1})
    {
        decoder->seek(static_cast<double>(frame) / rate, aborter);
        if (frame >= frames)
        {
            require(!decoder->run(chunk, aborter), "seek past EOF returned samples");
        }
        else
        {
            require(decoder->run(chunk, aborter), "seek returned premature EOF");
            require(chunk.get_srate() == rate && chunk.get_channels() == channels,
                    "chunk format differs");
            const auto offset = frame * channels;
            require(offset + chunk.get_data_length() <= expected.size(), "seek chunk exceeds EOF");
            require(std::equal(chunk.get_data(), chunk.get_data() + chunk.get_data_length(),
                               expected.begin() + offset),
                    "sample-inaccurate seek");
        }

        decoder->get_info(0, info, aborter);
    }
    decoder->initialize(0, input_flag_playback, aborter);

    for (size_t frame : {frames / 2, frames - 1, size_t(1), size_t(0)})
    {
        decoder->seek(static_cast<double>(frame) / rate, aborter);
        require(decoder->run(chunk, aborter), "playback seek returned premature EOF");
        const auto offset = frame * channels;
        require(offset + chunk.get_data_length() <= expected.size(), "playback chunk exceeds EOF");
        require(std::equal(chunk.get_data(), chunk.get_data() + chunk.get_data_length(),
                           expected.begin() + offset),
                "native playback seek changed PCM");
    }
    decoder->initialize(0, input_flag_playback, aborter);
    require(read_all(decoder) == expected, "reinitializing changed PCM");

    auto closing_decoder = open_decoder(la);
    closing_decoder->initialize(0, input_flag_playback, aborter);
    closing_decoder.release();
    auto cancel_decoder = open_decoder(la);
    CancelOnSecondCheck cancel;
    bool aborted = false;
    try
    {
        cancel_decoder->run(chunk, cancel);
    }
    catch (const exception_aborted &)
    {
        aborted = true;
    }
    require(aborted, "abort exception was not preserved across Rust boundary");
    log << "PASS " << pfc::stringcvt::string_utf8_from_wide(la.filename().c_str()).get_ptr() << ": "
        << frames << " sample frames, PCM + seeking + reinitialize\n";
    log.flush();
}
void run_tests(const std::filesystem::path &root)
{
    std::ofstream log(root / (sizeof(void *) == 8 ? "foobar2000/build/host-test-results-x64.txt"
                                                  : "foobar2000/build/host-test-results-Win32.txt"),
                      std::ios::trunc);
    log << "foobar2000 host decoder integration tests\n";
    log.flush();
    try
    {
        size_t count = 0;
        for (const auto *name :
             {"silence_mono", "edges_mono", "noise_stereo", "tone_stereo", "multiframe_mono"})
        {
            const auto fixtures = root / "tests/fixtures";
            for (const auto *mode : {"normal", "high", "noseek", "high_noseek"})
            {
                check_file(fixtures / (std::string(name) + "." + mode + ".la"),
                           fixtures / (std::string(name) + ".wav"), log);
                ++count;
            }
        }
        const auto test_dir = root / (sizeof(void *) == 8 ? "foobar2000/build/host-fixtures-x64"
                                                          : "foobar2000/build/host-fixtures-Win32");
        std::filesystem::create_directories(test_dir);
        const auto fixture = root / "tests/fixtures/edges_mono.normal.la";
        const auto tagged = test_dir / "tagged.la";
        std::filesystem::copy_file(fixture, tagged,
                                   std::filesystem::copy_options::overwrite_existing);
        pfc::string8 canonical;
        filesystem::g_get_canonical_path(pfc::stringcvt::string_utf8_from_wide(tagged.c_str()),
                                         canonical);
        file::ptr tag_file;
        filesystem::g_open(tag_file, canonical, filesystem::open_mode_write_existing, aborter);
        file_info_impl tags;
        tags.meta_set("TITLE", "La Rust playback test");
        tag_processor::write_apev2(tag_file, tags, aborter);
        tag_file.release();
        check_file(tagged, root / "tests/fixtures/edges_mono.wav", log);
        auto tagged_decoder = open_decoder(tagged);
        file_info_impl tagged_info;
        tagged_decoder->get_info(0, tagged_info, aborter);
        require(std::string(tagged_info.meta_get("TITLE", 0)) == "La Rust playback test",
                "APEv2 title was lost");
        log << "PASS APEv2 metadata\n";
        ++count;
        std::ifstream source(fixture, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(source)), {});
        bytes[45] ^= 1;
        const auto damaged = test_dir / "bad-crc.la";
        {
            std::ofstream out(damaged, std::ios::binary);
            out.write(bytes.data(), bytes.size());
        }
        bool rejected = false;
        try
        {
            read_all(open_decoder(damaged));
        }
        catch (const exception_io_data &)
        {
            rejected = true;
        }
        require(rejected, "bad PCM CRC was accepted");
        rejected = false;
        try
        {
            auto playback = open_decoder(damaged);
            playback->initialize(0, input_flag_playback, aborter);
            read_all(playback);
        }
        catch (const exception_io_data &)
        {
            rejected = true;
        }
        require(rejected, "native playback swallowed the CRC error");
        log << "PASS bad CRC raises foobar2000 data error\n";

        if (std::filesystem::exists(root / "local/audio"))
        {
            for (const auto &entry : std::filesystem::directory_iterator(root / "local/audio"))
            {
                if (entry.path().extension() == L".la")
                {
                    auto wav = entry.path();
                    wav.replace_extension(L".wav");
                    if (std::filesystem::exists(wav))
                    {
                        check_file(entry.path(), wav, log);
                        ++count;
                    }
                }
            }
        }
        log << "SUCCESS: " << count << " files passed inside foobar2000.\n";
    }
    catch (const std::exception &e)
    {
        log << "FAIL: " << e.what() << "\n";
    }
    catch (...)
    {
        log << "FAIL: unknown exception\n";
    }
}
void run_seek_benchmark(const std::filesystem::path &root)
{
    std::ofstream log(root / (sizeof(void *) == 8 ? "foobar2000/build/seek-benchmark-x64.txt"
                                                  : "foobar2000/build/seek-benchmark-Win32.txt"));
    log << std::fixed << std::setprecision(3);
    try
    {
        if (std::filesystem::exists(root / "local/audio"))
        {
            for (const auto &entry : std::filesystem::directory_iterator(root / "local/audio"))
            {
                if (entry.path().extension() != L".la")
                    continue;
                auto wav = entry.path();
                wav.replace_extension(L".wav");
                if (!std::filesystem::exists(wav))
                    continue;
                auto reference = open_decoder(wav);
                file_info_impl info;
                reference->get_info(0, info, aborter);
                auto expected = read_all(reference);
                const auto channels = static_cast<size_t>(info.info_get_int("channels"));
                const auto rate = static_cast<unsigned>(info.info_get_int("samplerate"));
                const auto frames = expected.size() / channels;
                auto decoder = open_decoder(entry.path());
                const auto start = std::chrono::steady_clock::now();
                decoder->initialize(0, input_flag_playback, aborter);
                audio_chunk_impl chunk;
                require(decoder->run(chunk, aborter), "no first playback chunk");
                log << "first_chunk_ms="
                    << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                 start)
                           .count()
                    << "\n";
                for (unsigned percent : {75, 25, 90, 10, 50})
                {
                    const auto target = frames * percent / 100;
                    const auto before = std::chrono::steady_clock::now();
                    decoder->seek(static_cast<double>(target) / rate, aborter);
                    require(decoder->run(chunk, aborter), "seek produced no PCM");
                    const auto elapsed = std::chrono::duration<double, std::milli>(
                                             std::chrono::steady_clock::now() - before)
                                             .count();
                    require(std::equal(chunk.get_data(), chunk.get_data() + chunk.get_data_length(),
                                       expected.begin() + target * channels),
                            "benchmark seek changed PCM");
                    log << "seek_" << percent << "_percent_ms=" << elapsed << "\n";
                    log.flush();
                }
                log << "SUCCESS: all benchmark seeks preserve exact PCM.\n";
            }
        }
    }
    catch (const std::exception &e)
    {
        log << "FAIL: " << e.what() << "\n";
    }
}
class HostTest : public initquit
{
    void on_init() override
    {
        wchar_t path[32768] = {};
        if (GetEnvironmentVariableW(L"LA04_TEST_ROOT", path, 32768) == 0)
            return;
        const std::filesystem::path root(path);

        wchar_t benchmark[2] = {};
        const bool bench = GetEnvironmentVariableW(L"LA04_SEEK_BENCH", benchmark, 2) != 0;
        fb2k::inMainThread([root, bench] {
            if (bench)
                run_seek_benchmark(root);
            else
                run_tests(root);
            standard_commands::main_exit();
        });
    }
};
initquit_factory_t<HostTest> factory;
}
