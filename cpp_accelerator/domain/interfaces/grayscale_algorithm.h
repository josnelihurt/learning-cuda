#pragma once

namespace jrb::domain::interfaces {

enum class GrayscaleAlgorithm {
  BT601,      // ITU-R BT.601 (SDTV): Y = 0.299R + 0.587G + 0.114B
  BT709,      // ITU-R BT.709 (HDTV): Y = 0.2126R + 0.7152G + 0.0722B
  Average,    // Simple average: Y = (R + G + B) / 3
  Lightness,  // Lightness: Y = (max(R,G,B) + min(R,G,B)) / 2
  Luminosity  // Luminosity: Y = 0.21R + 0.72G + 0.07B
};

}  // namespace jrb::domain::interfaces
