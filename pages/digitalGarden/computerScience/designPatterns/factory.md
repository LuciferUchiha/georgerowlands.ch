# Factory

Factory patterns are so called creational patterns meaning their intent is to abstract/hide how objects are created. Which in return allows the client to be in independent of how its objects are created.

## Factory method

The factory method pattern delegates the instantiation of objects to a method in either subclasses or a static method. The disadvantage of having the factory method as a static method is it can not be subclassed to change the behavior however you don't need to create an object to make use of the method.

### Structure

```mermaid
classDiagram
  Creator <|-- ConcreteCreator
  Creator --> ProductInterface
  ProductInterface <|-- ConcreteProduct
  class Creator{
    +someOperation()
    +createProduct(): Product
  }
  class ProductInterface{
  }
  class ConcreteProduct{
  }
  class ConcreteCreator{
    +createProduct(): Product 
 returns a ConcreteProduct
  } 
```

### Example

![factoryMethod](/img/programming/factoryMethod.png)

## Abstract factory

The factory method pattern delegates the instantiation of object familys to a another object.

### Structure

![abstractFactoryStructure](/img/programming/abstractFactoryStructure.png)

A big question here is where is the concrete Factory so they can all have acces to it. Often this is done in it's own class

```java
public class CurrentFactory {
 private CurrentFactory() { }; // prevents instantiation
 private static Factory fac = null;
 public static Factory getFactory() { return fac; }
 public static void setFactory(Factory f) {
  if (f == null) throw new NullPointerException();
  fac = f;
 }
}
```

### Example

![abstractFactory](/img/programming/abstractFactory.png)
