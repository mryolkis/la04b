#include "la_bridge.h"
#include <cmath>
#include <exception>
#include <foobar2000/SDK/foobar2000.h>
#include <memory>
#include <utility>

DECLARE_COMPONENT_VERSION(
    "La Rust Decoder", "0.1.1",
    "La 0.4b (LA04) lossless audio decoder.\n"
    "Native Rust decoding, 16-bit mono/stereo, normal/high, sample-accurate seeking.\n"
    "Based on the reconstructed la04 Rust codec.\n"
    "Original La format by Michael Bevin (2002-2004).\n"
    "Built with foobar2000 SDK 2026-10-01.");
VALIDATE_COMPONENT_FILENAME("foo_input_la.dll");

class input_la : public input_stubs
{
  public:
    void open(file::ptr hint, const char *path, t_input_open_reason reason, abort_callback &abort)
    {
        if (reason == input_open_info_write)
            throw exception_tagging_unsupported();
        m_file = hint;
        input_open_file_helper(m_file, path, reason, abort);
        m_file->ensure_seekable();
        m_size = m_file->get_size_ex(abort);
        tag_processor::read_trailing_nothrow(m_file, m_tags, abort);
        m_abort = &abort;
        char error[256] = {};
        m_decoder.reset(
            la_open({this, read_callback, seek_callback, m_size, check_callback}, &m_info, error));
        check(m_decoder ? 0 : -1, error);
        m_abort = nullptr;
    }

    void get_info(file_info &info, abort_callback &abort)
    {
        abort.check();
        info = m_tags;
        const double length = static_cast<double>(m_info.sample_frames) / m_info.sample_rate;
        info.set_length(length);
        info.info_set("codec", "La 0.4b");
        info.info_set("encoding", "lossless");
        info.info_set("codec_profile", m_info.high ? "high" : "normal");
        info.info_set_int("samplerate", m_info.sample_rate);
        info.info_set_int("channels", m_info.channels);
        info.info_set_int("bitspersample", 16);
        info.info_set("la_seek_table", m_info.seek_table ? "yes" : "no");
        if (length > 0)
            info.info_set_bitrate(static_cast<unsigned>(m_size * 8.0 / length / 1000.0 + 0.5));
    }

    t_filestats get_file_stats(abort_callback &abort)
    {
        return m_file->get_stats(abort);
    }
    t_filestats2 get_stats2(unsigned flags, abort_callback &abort)
    {
        return m_file->get_stats2_(flags, abort);
    }

    void decode_initialize(unsigned, abort_callback &abort)
    {
        seek_sample(0, abort);
    }
    bool decode_run(audio_chunk &chunk, abort_callback &abort)
    {
        abort.check();
        m_abort = &abort;
        const std::int16_t *samples = nullptr;
        std::size_t count = 0;
        char error[256] = {};
        const auto result = la_read(m_decoder.get(), &samples, &count, error);
        check(result, error);
        m_abort = nullptr;
        if (result == 0)
            return false;
        chunk.set_data_fixedpoint(samples, count * sizeof(*samples), m_info.sample_rate,
                                  m_info.channels, 16,
                                  audio_chunk::g_guess_channel_config(m_info.channels));
        return true;
    }
    void decode_seek(double seconds, abort_callback &abort)
    {
        if (!std::isfinite(seconds) || seconds < 0)
            throw exception_io_data("Invalid seek time");
        const double duration = static_cast<double>(m_info.sample_frames) / m_info.sample_rate;
        const auto target = seconds >= duration
                                ? m_info.sample_frames
                                : audio_math::time_to_samples(seconds, m_info.sample_rate);
        seek_sample(target, abort);
    }
    bool decode_can_seek()
    {
        return true;
    }
    bool decode_get_dynamic_info(file_info &, double &)
    {
        return false;
    }
    bool decode_get_dynamic_info_track(file_info &, double &)
    {
        return false;
    }
    void decode_on_idle(abort_callback &abort)
    {
        m_file->on_idle(abort);
    }
    void retag(const file_info &, abort_callback &)
    {
        throw exception_tagging_unsupported();
    }
    void remove_tags(abort_callback &)
    {
        throw exception_tagging_unsupported();
    }

    static bool g_is_our_content_type(const char *)
    {
        return false;
    }
    static bool g_is_our_path(const char *, const char *extension)
    {
        return stricmp_utf8(extension, "la") == 0;
    }
    static const char *g_get_name()
    {
        return "La Rust Decoder";
    }
    static GUID g_get_guid()
    {
        return {0x436922c8, 0xa722, 0x46c1, {0x9b, 0xc3, 0xa2, 0x08, 0x7d, 0x24, 0x7d, 0x2a}};
    }

  private:
    static std::int32_t check_callback(void *context) noexcept
    {
        auto &self = *static_cast<input_la *>(context);
        try
        {
            self.m_abort->check();
            return 0;
        }
        catch (...)
        {
            self.m_exception = std::current_exception();
            return -1;
        }
    }
    static std::intptr_t read_callback(void *context, std::uint8_t *buffer,
                                       std::size_t size) noexcept
    {
        auto &self = *static_cast<input_la *>(context);
        try
        {
            self.m_abort->check();
            return static_cast<std::intptr_t>(self.m_file->read(buffer, size, *self.m_abort));
        }
        catch (...)
        {
            self.m_exception = std::current_exception();
            return -1;
        }
    }
    static std::int32_t seek_callback(void *context, std::uint64_t position) noexcept
    {
        auto &self = *static_cast<input_la *>(context);
        try
        {
            self.m_abort->check();
            self.m_file->seek(position, *self.m_abort);
            return 0;
        }
        catch (...)
        {
            self.m_exception = std::current_exception();
            return -1;
        }
    }
    void check(std::int32_t result, const char *error)
    {
        if (m_exception)
        {
            auto saved = std::exchange(m_exception, {});
            std::rethrow_exception(saved);
        }
        if (result < 0)
            throw exception_io_data(error);
    }
    void seek_sample(std::uint64_t target, abort_callback &abort)
    {
        abort.check();
        m_abort = &abort;
        char error[256] = {};
        check(la_seek(m_decoder.get(), target, error), error);
        m_abort = nullptr;
    }
    file::ptr m_file;
    file_info_impl m_tags;
    LaInfo m_info = {};
    std::uint64_t m_size = 0;
    abort_callback *m_abort = nullptr;
    std::exception_ptr m_exception;
    std::unique_ptr<LaDecoder, decltype(&la_close)> m_decoder{nullptr, la_close};
};

static input_singletrack_factory_t<input_la> g_input_la;
DECLARE_FILE_TYPE("La lossless audio", "*.LA");
