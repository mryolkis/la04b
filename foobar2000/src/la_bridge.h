#pragma once
#include <cstddef>
#include <cstdint>

extern "C"
{
    struct LaIo
    {
        void *context;
        std::intptr_t (*read)(void *, std::uint8_t *, std::size_t);
        std::int32_t (*seek)(void *, std::uint64_t);
        std::uint64_t size;
        std::int32_t (*check)(void *);
    };
    struct LaInfo
    {
        std::uint64_t sample_frames;
        std::uint32_t sample_rate;
        std::uint16_t channels;
        std::uint8_t high;
        std::uint8_t seek_table;
    };
    struct LaDecoder;

    LaDecoder *la_open(LaIo io, LaInfo *info, char *err);

    std::int32_t la_read(LaDecoder *, const std::int16_t **samples, std::size_t *count, char *err);
    std::int32_t la_seek(LaDecoder *, std::uint64_t sample_frame, char *err);
    void la_close(LaDecoder *);
}
