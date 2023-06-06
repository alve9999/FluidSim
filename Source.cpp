#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#define nx 1024
#define ny 1024

void CUDA_INIT();
void CUDA_EXIT();
void COMPUTE_FIELD(uint8_t* result);



int main() {
    CUDA_INIT();

    sf::RenderWindow window(sf::VideoMode(nx, ny), "FluidSim");

    sf::Texture texture;
    sf::Sprite sprite;

    //RGBA32 vector
    sf::Uint8* pixelBuffer;
    pixelBuffer = (sf::Uint8*)malloc(4*nx * ny * sizeof(sf::Uint8*));
    texture.create(ny, nx);


    while (window.isOpen()) {
        COMPUTE_FIELD(pixelBuffer);
        texture.update(pixelBuffer);
        sprite.setTexture(texture);
        window.draw(sprite);
        window.display();
    }

    CUDA_EXIT();
}