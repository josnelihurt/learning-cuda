package steps

import (
	"fmt"
)

func (tc *TestContext) eachVideoShouldHaveNonEmptyID() error {
	videos, err := tc.GetVideosFromResponse()
	if err != nil {
		return err
	}
	for _, vid := range videos {
		if vid.Id == "" {
			return fmt.Errorf("video has empty id")
		}
	}
	return nil
}

func (tc *TestContext) eachVideoShouldHaveNonEmptyDisplayName() error {
	videos, err := tc.GetVideosFromResponse()
	if err != nil {
		return err
	}
	for _, vid := range videos {
		if vid.DisplayName == "" {
			return fmt.Errorf("video %s has empty display name", vid.Id)
		}
	}
	return nil
}

func (tc *TestContext) eachVideoShouldHaveNonEmptyPath() error {
	videos, err := tc.GetVideosFromResponse()
	if err != nil {
		return err
	}
	for _, vid := range videos {
		if vid.Path == "" {
			return fmt.Errorf("video %s has empty path", vid.Id)
		}
	}
	return nil
}

func (tc *TestContext) eachVideoShouldHaveNonEmptyPreviewImagePath() error {
	videos, err := tc.GetVideosFromResponse()
	if err != nil {
		return err
	}
	for _, vid := range videos {
		if vid.PreviewImagePath == "" {
			return fmt.Errorf("video %s has empty preview image path", vid.Id)
		}
	}
	return nil
}

func (tc *TestContext) atLeastOneVideoShouldBeMarkedAsDefault() error {
	videos, err := tc.GetVideosFromResponse()
	if err != nil {
		return err
	}
	for _, vid := range videos {
		if vid.IsDefault {
			return nil
		}
	}
	return fmt.Errorf("no video marked as default")
}
