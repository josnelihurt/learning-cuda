package main

import (
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"image"
	_ "image/png"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
)

type ChecksumEntry struct {
	Image         string `json:"image"`
	Filter        string `json:"filter"`
	Accelerator   string `json:"accelerator"`
	GrayscaleType string `json:"grayscale_type,omitempty"`
	Checksum      string `json:"checksum"`
	Width         int32  `json:"width"`
	Height        int32  `json:"height"`
	Channels      int32  `json:"channels"`
}

type ChecksumData struct {
	GeneratedAt string          `json:"generated_at"`
	Checksums   []ChecksumEntry `json:"checksums"`
}

func main() {
	if len(os.Args) < 2 {
		log.Fatal("Usage: go run generate_checksums.go <service-url>")
	}

	serviceURL := os.Args[1]
	testdataDir := "../testdata"
	dataDir := "../../../../data"
	outputFile := filepath.Join(testdataDir, "checksums.json")

	client := newImageProcessorClient(serviceURL)
	checksumData := ChecksumData{
		GeneratedAt: time.Now().Format(time.RFC3339),
		Checksums:   []ChecksumEntry{},
	}

	images := []string{"lena.png"}
	filters := []pb.FilterType{pb.FilterType_FILTER_TYPE_NONE, pb.FilterType_FILTER_TYPE_GRAYSCALE}
	accelerators := []pb.AcceleratorType{pb.AcceleratorType_ACCELERATOR_TYPE_GPU, pb.AcceleratorType_ACCELERATOR_TYPE_CPU}
	grayscaleTypes := []pb.GrayscaleType{
		pb.GrayscaleType_GRAYSCALE_TYPE_BT601,
		pb.GrayscaleType_GRAYSCALE_TYPE_BT709,
		pb.GrayscaleType_GRAYSCALE_TYPE_AVERAGE,
		pb.GrayscaleType_GRAYSCALE_TYPE_LIGHTNESS,
		pb.GrayscaleType_GRAYSCALE_TYPE_LUMINOSITY,
	}

	for _, imageName := range images {
		imagePath := filepath.Join(dataDir, imageName)
		imageData, width, height, channels, err := loadImage(imagePath)
		if err != nil {
			log.Fatalf("Failed to load image %s: %v", imageName, err)
		}

		for _, filter := range filters {
			for _, accelerator := range accelerators {
				if filter == pb.FilterType_FILTER_TYPE_GRAYSCALE {
					for _, gsType := range grayscaleTypes {
						entry := processAndChecksum(
							client, imageName, imageData, width, height, channels,
							filter, accelerator, gsType,
						)
						if entry != nil {
							checksumData.Checksums = append(checksumData.Checksums, *entry)
							log.Printf("[OK] %s + %s + %s + %s = %s",
								imageName, filter, accelerator, gsType, entry.Checksum[:16]+"...")
						}
					}
				} else {
					entry := processAndChecksum(
						client, imageName, imageData, width, height, channels,
						filter, accelerator, pb.GrayscaleType_GRAYSCALE_TYPE_UNSPECIFIED,
					)
					if entry != nil {
						checksumData.Checksums = append(checksumData.Checksums, *entry)
						log.Printf("[OK] %s + %s + %s = %s",
							imageName, filter, accelerator, entry.Checksum[:16]+"...")
					}
				}
			}
		}
	}

	jsonData, err := json.MarshalIndent(checksumData, "", "  ")
	if err != nil {
		log.Fatalf("Failed to marshal JSON: %v", err)
	}

	if err := os.WriteFile(outputFile, jsonData, 0644); err != nil {
		log.Fatalf("Failed to write checksums file: %v", err)
	}

	log.Printf("\nGenerated %d checksums in %s", len(checksumData.Checksums), outputFile)
}

func newImageProcessorClient(baseURL string) genconnect.ImageProcessorServiceClient {
	httpClient := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
		Timeout: 30 * time.Second,
	}
	return genconnect.NewImageProcessorServiceClient(httpClient, baseURL)
}

func loadImage(path string) ([]byte, int32, int32, int32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, 0, 0, 0, err
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	rawData := make([]byte, width*height*4)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, a := img.At(x, y).RGBA()
			idx := (y*width + x) * 4
			rawData[idx] = byte(r >> 8)
			rawData[idx+1] = byte(g >> 8)
			rawData[idx+2] = byte(b >> 8)
			rawData[idx+3] = byte(a >> 8)
		}
	}

	return rawData, int32(width), int32(height), int32(4), nil
}

func processAndChecksum(
	client genconnect.ImageProcessorServiceClient,
	imageName string,
	imageData []byte,
	width, height, channels int32,
	filter pb.FilterType,
	accelerator pb.AcceleratorType,
	grayscaleType pb.GrayscaleType,
) *ChecksumEntry {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := &pb.ProcessImageRequest{
		ImageData:     imageData,
		Width:         width,
		Height:        height,
		Channels:      channels,
		Filters:       []pb.FilterType{filter},
		Accelerator:   accelerator,
		GrayscaleType: grayscaleType,
	}

	resp, err := client.ProcessImage(ctx, connect.NewRequest(req))
	if err != nil {
		log.Printf("Failed to process image: %v", err)
		return nil
	}

	if resp.Msg.Code != 0 {
		log.Printf("Processing failed: %s", resp.Msg.Message)
		return nil
	}

	checksum := calculateChecksum(resp.Msg.ImageData)

	entry := &ChecksumEntry{
		Image:       imageName,
		Filter:      filter.String(),
		Accelerator: accelerator.String(),
		Checksum:    checksum,
		Width:       resp.Msg.Width,
		Height:      resp.Msg.Height,
		Channels:    resp.Msg.Channels,
	}

	if grayscaleType != pb.GrayscaleType_GRAYSCALE_TYPE_UNSPECIFIED {
		entry.GrayscaleType = grayscaleType.String()
	}

	return entry
}

func calculateChecksum(data []byte) string {
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}
