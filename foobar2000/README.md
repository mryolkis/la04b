# La для foobar2000

Компонент `foo_input_la` воспроизводит LA04 в foobar2000 2.x для Windows 10/11.
C++ адаптер обращается к Rust-декодеру через статически включённый FFI-мост.

Поддерживаются PCM 16 бит, mono/stereo, normal/high, точная перемотка,
чтение APE/ID3v1-тегов и ReplayGain средствами плеера. Источник должен
предоставлять размер и позиционирование. Запись тегов, живые потоки,
старые версии La и Windows ARM64 не поддерживаются.

Поведение перемотки, расход памяти и проверка CRC описаны в
[документации кодека](../docs/codec.md).

## Сборка

Нужны rustup, Visual Studio 2022 Build Tools с C++, Windows SDK и 7-Zip.
Rust 1.100.0 выбирается файлом `../rust-toolchain.toml`.

Из PowerShell в корне проекта:

```powershell
.\foobar2000\build.ps1
```

Скрипт устанавливает MSVC targets для x86 и x64, собирает Rust staticlib,
C++ адаптер и создаёт `dist/foo_input_la.fb2k-component` относительно этой
папки. При необходимости он загружает официальный SDK foobar2000 в `vendor/sdk`
и проверяет SHA-256 архива. Первая сборка требует доступа к интернету.

Для одной архитектуры укажите `-Platform x64` или `-Platform Win32`.
DLL находятся в `build/x64` и `build/Win32`, Rust staticlib — в общем
`../target/<target>/release`.

## Установка

1. В foobar2000 откройте `File → Preferences → Components`.
2. Нажмите `Install…` и выберите `dist/foo_input_la.fb2k-component`.
3. Нажмите `Apply` и перезапустите плеер.

Общий пакет содержит обе архитектуры; плеер выберет подходящую DLL.
Если установлен старый компонент `foo_la`, удалите его через список компонентов,
чтобы избежать конкуренции за расширение `.la`.

## Проверка в плеере

После сборки, из корня проекта:

```powershell
.\foobar2000\tests\run_host_test.ps1
.\foobar2000\tests\run_host_test.ps1 -Platform Win32 -FoobarPath 'C:\Program Files (x86)\foobar2000'
.\foobar2000\tests\run_host_test.ps1 -SeekBenchmark
```

По умолчанию используется x64-плеер из `C:\Program Files\foobar2000`.
`-FoobarPath` позволяет указать другой каталог установки. Скрипт создаёт
отдельную portable-копию в `build/test-player-<platform>` и тестовый компонент.
Проверяются эталонные файлы из `../tests/fixtures`, теги, CRC, отмена,
повторная инициализация и точность перемотки. При наличии пар `.la`/`.wav`
в `../local/audio` они также проходят проверку.

`-SeekBenchmark` измеряет время переходов по записи из `../local/audio`.
`-DecoderPath` задаёт DLL для сравнения с другой сборкой. Отчёты сохраняются
в `build/host-test-results-<platform>.txt` и `build/seek-benchmark-<platform>.txt`.
Тестовый компонент в установочный пакет не входит.

## Контракт FFI

ABI объявлен в `src/la_bridge.h`, реализация находится в `rust/src/lib.rs`.
Вызовы синхронны; один handle доступен только одному потоку одновременно.

- I/O callbacks и их context действуют до `la_close`; C++ исключения должны
  перехватываться callback-функциями и преобразовываться в код ошибки.
- Буфер `err` должен содержать не менее 256 доступных для записи байт.
- `la_open` получает доступный для записи `LaInfo`.
- `la_read` возвращает 1 для PCM, 0 для EOF, -1 при ошибке. `samples` и `count`
  должны быть доступны для записи. `count` измеряется в чередующихся `i16`.
- PCM принадлежит декодеру и действителен до следующего read, seek или close.
- `la_seek` принимает позицию в отсчётах на канал.
- `la_close` принимает null или действующий handle, который больше не используется.
- После ошибки handle закрывается; для продолжения декодер открывается заново.
